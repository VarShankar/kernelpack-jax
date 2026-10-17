// Optional native cuDSS backend. NVIDIA headers/libraries are external dependencies.
#include "xla/ffi/api/ffi.h"
#include <cudss.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <tuple>
#include <type_traits>

namespace ffi = xla::ffi;
static void Check(cudssStatus_t s) {
  if (s != CUDSS_STATUS_SUCCESS) throw std::runtime_error("cuDSS status " + std::to_string(s));
}
static void CheckCuda(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct DeviceGuard {
  int previous;
  explicit DeviceGuard(int device) { CheckCuda(cudaGetDevice(&previous)); CheckCuda(cudaSetDevice(device)); }
  ~DeviceGuard() { cudaSetDevice(previous); }
};
static int AsyncAlloc(void*, void **p, size_t n, cudaStream_t s) { return cudaMallocAsync(p,n,s)==cudaSuccess ? 0 : 1; }
static int AsyncFree(void*, void *p, size_t, cudaStream_t s) { return cudaFreeAsync(p,s)==cudaSuccess ? 0 : 1; }
struct Factor {
  struct GraphSolve {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    cudssMatrix_t b=nullptr,x=nullptr;
    void clear() noexcept {
      if(executable) cudaGraphExecDestroy(executable);
      if(graph) cudaGraphDestroy(graph);
      if(b) cudssMatrixDestroy(b);
      if(x) cudssMatrixDestroy(x);
      executable=nullptr; graph=nullptr; b=nullptr; x=nullptr;
    }
  };
  bool graph_enabled=false;
  int64_t graph_launches=0;
  std::unordered_map<int64_t,GraphSolve> graphs;
  struct Pending {
    cudaEvent_t ready = nullptr, done = nullptr;
    cudssMatrix_t b = nullptr, x = nullptr;
    int64_t columns = 0;
  };
  int device = 0;
  int reordering = 0;  // Stable API selector: default=0, none=1, amd=2.
  int64_t n = 0;
  int64_t nnz = 0;
  cudssDataType_t dtype = CUDSS_R_64F;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t a = nullptr;
  cudssMatrix_t b = nullptr, x = nullptr;
  void *rhs_storage = nullptr, *solution_storage = nullptr;
  int64_t rhs_columns = 0;
  void *rows = nullptr, *cols = nullptr, *values = nullptr;
  std::mutex mutex;
  int64_t solves = 0, descriptors_created = 0, descriptors_reused = 0;
  int64_t refactorizations = 0;
  int64_t analyses = 0;
  // Only immutable setup-time matrix variants may use this cache key.
  int64_t variant = -1;
  cudaStream_t solve_stream = nullptr;
  std::vector<Pending> pending, available;
  void retire(bool wait) {
    if (wait && solve_stream) CheckCuda(cudaStreamSynchronize(solve_stream));
    for (auto it=pending.begin(); it!=pending.end();) {
      auto status=cudaEventQuery(it->done);
      if (status==cudaErrorNotReady) { ++it; continue; }
      CheckCuda(status);
      available.push_back(*it);
      it=pending.erase(it);
    }
  }
  ~Factor() {
    // Destructors must not throw, including during interpreter/CUDA shutdown.
    int previous=-1; cudaGetDevice(&previous);
    if (cudaSetDevice(device)!=cudaSuccess) return;
    if (solve_stream) cudaStreamSynchronize(solve_stream);
    for (auto &p:pending) {
      cudssMatrixDestroy(p.b); cudssMatrixDestroy(p.x);
      cudaEventDestroy(p.ready); cudaEventDestroy(p.done);
    }
    for (auto &p:available) {
      cudssMatrixDestroy(p.b); cudssMatrixDestroy(p.x);
      cudaEventDestroy(p.ready); cudaEventDestroy(p.done);
    }
    for (auto &entry:graphs) {
      auto &g=entry.second;
      g.clear();
    }
    if (data) cudssDataDestroy(handle, data);
    // The custom device_free callback enqueues cudaFreeAsync. Observe those
    // frees before this factor's stream/handle go away or a rebuild allocates
    // the next factor. Keep destructor cleanup non-throwing.
    if (solve_stream) cudaStreamSynchronize(solve_stream);
    if (a) cudssMatrixDestroy(a);
    if (b) cudssMatrixDestroy(b);
    if (x) cudssMatrixDestroy(x);
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
    cudaFree(rows); cudaFree(cols); cudaFree(values);
    cudaFree(rhs_storage); cudaFree(solution_storage);
    if (solve_stream) cudaStreamDestroy(solve_stream);
    if (previous>=0) cudaSetDevice(previous);
  }
  void check_info() {
    int info = 0; size_t written = 0;
    Check(cudssDataGet(handle, data, CUDSS_DATA_INFO, &info, sizeof(info), &written));
    if (info) throw std::runtime_error("cuDSS numerical failure, info=" + std::to_string(info));
  }
};
static std::mutex registry_mutex;
static std::unordered_map<int64_t, std::shared_ptr<Factor>> registry;
static std::unordered_map<int64_t,bool> rebuilding_contexts;
static std::atomic<int64_t> next_id{1};
static thread_local std::string last_error;
extern "C" const char *kp_cudss_error() { return last_error.c_str(); }

// Stable public selector values are independent of NVIDIA enum ordinals.
static cudssReorderingAlg_t ReorderingAlgorithm(int selector) {
  switch(selector) {
    case 0: return CUDSS_REORDERING_ALG_DEFAULT;
    case 1: return CUDSS_REORDERING_ALG_NONE;
    case 2: return CUDSS_REORDERING_ALG_AMD;
    default: throw std::runtime_error("cuDSS reordering must be default, none, or amd");
  }
}

// Host setup: inputs are synchronized GPU pointers. Own CSR storage so donation
// or releasing the Python input arrays cannot invalidate the factor.
static int64_t CreateContext(int device, int64_t n, int64_t nnz,
    const void *rp, const void *ci, const void *av, int is_double, int spd, int64_t rhs_columns,
    const char *threading_lib, int reordering) {
  try {
    const auto algorithm=ReorderingAlgorithm(reordering);
    DeviceGuard guard(device);
    auto f = std::make_shared<Factor>(); f->device=device; f->n=n; f->nnz=nnz;
    f->reordering=reordering;
    f->rhs_columns=rhs_columns;
    f->dtype = is_double ? CUDSS_R_64F : CUDSS_R_32F;
    CheckCuda(cudaMalloc(&f->rows, (n+1)*sizeof(int32_t)));
    CheckCuda(cudaMalloc(&f->cols, nnz*sizeof(int32_t)));
    CheckCuda(cudaMalloc(&f->values, nnz*(is_double ? sizeof(double) : sizeof(float))));
    CheckCuda(cudaMemcpy(f->rows,rp,(n+1)*sizeof(int32_t),cudaMemcpyDeviceToDevice));
    CheckCuda(cudaMemcpy(f->cols,ci,nnz*sizeof(int32_t),cudaMemcpyDeviceToDevice));
    CheckCuda(cudaMemcpy(f->values,av,nnz*(is_double ? sizeof(double) : sizeof(float)),cudaMemcpyDeviceToDevice));
    Check(cudssCreate(&f->handle));
    cudssDeviceMemHandler_t allocator{};
    allocator.device_alloc=AsyncAlloc; allocator.device_free=AsyncFree;
    Check(cudssSetDeviceMemHandler(f->handle,&allocator));
    CheckCuda(cudaStreamCreateWithFlags(&f->solve_stream,cudaStreamNonBlocking));
    Check(cudssSetStream(f->handle,f->solve_stream));
    if (threading_lib && threading_lib[0]) Check(cudssSetThreadingLayer(f->handle,threading_lib));
    Check(cudssConfigCreate(&f->config)); Check(cudssDataCreate(f->handle,&f->data));
    // Preserve the prior vendor-default path exactly when selector is zero.
    if (reordering!=0)
      Check(cudssConfigSet(f->config,CUDSS_CONFIG_REORDERING_ALG,&algorithm,sizeof(algorithm)));
    if (!spd) {
      auto matching = CUDSS_MATCHING_ALG_AUTO;
      Check(cudssConfigSet(f->config,CUDSS_CONFIG_MATCHING_ALG,&matching,sizeof(matching)));
      int refinement_steps = 2;
      Check(cudssConfigSet(f->config,CUDSS_CONFIG_IR_N_STEPS,&refinement_steps,sizeof(refinement_steps)));
    }
    Check(cudssMatrixCreateCsr(&f->a,n,n,nnz,f->rows,nullptr,f->cols,f->values,
      CUDSS_R_32I,CUDSS_R_32I,f->dtype,spd ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_GENERAL,
      CUDSS_MVIEW_FULL,CUDSS_BASE_ZERO));
    // Plan workspace for the maximum RHS block. Actual solves create
    // per-call descriptors for their exact width and do not pad RHS columns.
    size_t dense_bytes=n*rhs_columns*(is_double ? sizeof(double) : sizeof(float));
    CheckCuda(cudaMalloc(&f->rhs_storage,dense_bytes));
    CheckCuda(cudaMalloc(&f->solution_storage,dense_bytes));
    CheckCuda(cudaMemset(f->rhs_storage,0,dense_bytes));
    CheckCuda(cudaMemset(f->solution_storage,0,dense_bytes));
    CheckCuda(cudaStreamSynchronize(nullptr));
    Check(cudssMatrixCreateDn(&f->b,n,rhs_columns,n,f->rhs_storage,f->dtype,CUDSS_LAYOUT_COL_MAJOR));
    Check(cudssMatrixCreateDn(&f->x,n,rhs_columns,n,f->solution_storage,f->dtype,CUDSS_LAYOUT_COL_MAJOR));
    Check(cudssExecute(f->handle,CUDSS_PHASE_ANALYSIS,f->config,f->data,f->a,f->x,f->b));
    ++f->analyses;
    Check(cudssExecute(f->handle,CUDSS_PHASE_FACTORIZATION,f->config,f->data,f->a,f->x,f->b));
    CheckCuda(cudaStreamSynchronize(f->solve_stream)); f->check_info();
    int64_t id=next_id.fetch_add(1);
    std::lock_guard<std::mutex> lock(registry_mutex); registry.emplace(id,std::move(f));
    return id;
  } catch (const std::exception &e) { last_error=e.what(); return -1; }
}
// Existing ABI retains its original default behavior.
extern "C" int64_t kp_cudss_create(int device, int64_t n, int64_t nnz,
    const void *rp, const void *ci, const void *av, int is_double, int spd,
    int64_t rhs_columns, const char *threading_lib) {
  return CreateContext(device,n,nnz,rp,ci,av,is_double,spd,rhs_columns,threading_lib,0);
}
extern "C" int64_t kp_cudss_create_with_reordering(int device, int64_t n, int64_t nnz,
    const void *rp, const void *ci, const void *av, int is_double, int spd,
    int64_t rhs_columns, const char *threading_lib, int reordering) {
  return CreateContext(device,n,nnz,rp,ci,av,is_double,spd,rhs_columns,threading_lib,reordering);
}

extern "C" void kp_cudss_destroy(int64_t id) {
  std::shared_ptr<Factor> f;
  { std::lock_guard<std::mutex> lock(registry_mutex);
    auto rebuilding=rebuilding_contexts.find(id);
    if (rebuilding!=rebuilding_contexts.end()) rebuilding->second=true;
    auto it=registry.find(id); if (it==registry.end()) return; f=it->second; registry.erase(it); }
  // In-flight FFI calls retain their own shared_ptr until completion.
}
extern "C" int64_t kp_cudss_solve_count(int64_t id) {
  std::shared_ptr<Factor> f;
  { std::lock_guard<std::mutex> lock(registry_mutex);
    auto it=registry.find(id); if (it==registry.end()) return -1; f=it->second; }
  std::lock_guard<std::mutex> lock(f->mutex); return f->solves;
}
extern "C" int kp_cudss_check_status(int64_t id) {
  try {
    std::shared_ptr<Factor> f;
    { std::lock_guard<std::mutex> lock(registry_mutex);
      auto it=registry.find(id); if (it==registry.end()) throw std::runtime_error("cuDSS factor is closed");
      f=it->second; }
    std::lock_guard<std::mutex> lock(f->mutex); DeviceGuard guard(f->device);
    f->retire(true); f->check_info(); return 0;
  } catch (const std::exception &e) { last_error=e.what(); return -1; }
}
extern "C" int kp_cudss_phase_counts(int64_t id, int64_t *analysis, int64_t *refactors) {
  std::shared_ptr<Factor> f;
  { std::lock_guard<std::mutex> lock(registry_mutex);
    auto it=registry.find(id); if(it==registry.end()) return -1; f=it->second; }
  std::lock_guard<std::mutex> lock(f->mutex);
  *analysis=f->analyses; *refactors=f->refactorizations; return 0;
}

// Eager changing-SPD setup. Numerical values stay on device; reuse symbolic
// analysis only when Python has verified that the complete CSR pattern agrees.
extern "C" int kp_cudss_refactor_values(int64_t id, const void *values) {
  try {
    std::shared_ptr<Factor> f;
    { std::lock_guard<std::mutex> lock(registry_mutex);
      auto it=registry.find(id); if(it==registry.end())
        throw std::runtime_error("cuDSS factor is closed");
      f=it->second; }
    std::lock_guard<std::mutex> lock(f->mutex); DeviceGuard guard(f->device);
    f->retire(true);
    for(auto &entry:f->graphs) entry.second.clear();
    f->graphs.clear();
    int clear_info=0;
    Check(cudssDataSet(f->handle,f->data,CUDSS_DATA_INFO,&clear_info,sizeof(clear_info)));
    CheckCuda(cudaMemcpyAsync(f->values,values,f->nnz*(f->dtype==CUDSS_R_64F ? sizeof(double) : sizeof(float)),
      cudaMemcpyDeviceToDevice,f->solve_stream));
    Check(cudssExecute(f->handle,CUDSS_PHASE_REFACTORIZATION,f->config,f->data,f->a,f->x,f->b));
    CheckCuda(cudaStreamSynchronize(f->solve_stream)); f->check_info();
    ++f->refactorizations; f->variant=-1;
    return 0;
  } catch(const std::exception &e) { last_error=e.what(); return -1; }
}

// Keep a training workspace's handle stable when its exact support changes.
// Already-enqueued calls own shared_ptrs; users must drain dependent results
// before this eager replacement. The old factor's destructor drains its stream.
extern "C" int kp_cudss_replace_context(int64_t destination, int64_t source) {
  std::shared_ptr<Factor> previous;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto target=registry.find(destination), replacement=registry.find(source);
    if(target==registry.end() || replacement==registry.end() || destination==source) return -1;
    if (target->second->reordering!=replacement->second->reordering) {
      last_error="cuDSS context replacement must preserve reordering";
      return -1;
    }
    previous=target->second;
    target->second=replacement->second;
    registry.erase(replacement);
  }
  return 0;
}

// Low-memory exact-pattern replacement for drained eager workspaces. Remove
// and destroy the old factor before allocating its replacement. A failed
// creation leaves this stable ID absent, so solves fail explicitly and a
// later rebuild can retry it without retaining an old numerical fallback.
static int RebuildContext(int64_t destination, int device,
    int64_t n, int64_t nnz, const void *rp, const void *ci, const void *av,
    int is_double, int spd, int64_t rhs_columns, const char *threading_lib, int reordering) {
  struct Reservation {
    int64_t destination;
    bool active=false;
    ~Reservation() {
      if (active) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        rebuilding_contexts.erase(destination);
      }
    }
  } reservation{destination};
  int64_t created_id=-1;
  try {
    // Reject invalid selectors before the registry changes or an old factor
    // is released. An absent-context retry still validates the explicit choice.
    (void)ReorderingAlgorithm(reordering);
    if (n<1 || nnz<1 || rhs_columns<1 || !rp || !ci || !av)
      throw std::runtime_error("Invalid cuDSS rebuild dimensions or device pointers");
    std::shared_ptr<Factor> previous;
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (destination<1 || destination>=next_id.load())
        throw std::runtime_error("cuDSS rebuild requires an allocated stable factor ID");
      if (rebuilding_contexts.count(destination))
        throw std::runtime_error("cuDSS context rebuild is already in progress");
      auto target=registry.find(destination);
      if (target!=registry.end()) {
        // The registry is the sole owner only after all host enqueue/status
        // calls have returned. Removing it under the same lock prevents a
        // new call from obtaining this factor after the ownership check.
        if (target->second.use_count()!=1)
          throw std::runtime_error("Drain concurrent cuDSS host calls before rebuilding");
        if (target->second->reordering!=reordering)
          throw std::runtime_error("cuDSS rebuild must preserve reordering");
        if (target->second->device!=device || target->second->n!=n
            || target->second->dtype!=(is_double ? CUDSS_R_64F : CUDSS_R_32F)
            || target->second->rhs_columns!=rhs_columns)
          throw std::runtime_error("cuDSS rebuild must preserve device, size, dtype and RHS capacity");
      }
      rebuilding_contexts.emplace(destination,false);
      reservation.active=true;
      if (target!=registry.end()) {
        previous=std::move(target->second);
        registry.erase(target);
      }
    }
    if (previous) {
      {
        DeviceGuard guard(previous->device);
        std::lock_guard<std::mutex> lock(previous->mutex);
        previous->retire(true);
      }
      // Destruction and all potentially expensive CUDA work occur without
      // holding the registry lock. No old factor survives the create below.
      previous.reset();
    }
    created_id=CreateContext(device,n,nnz,rp,ci,av,is_double,spd,rhs_columns,threading_lib,reordering);
    if (created_id<0) return -1;  // Preserve create's detailed last_error.
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      auto created=registry.find(created_id);
      auto rebuilding=rebuilding_contexts.find(destination);
      if (rebuilding==rebuilding_contexts.end() || rebuilding->second)
        throw std::runtime_error("cuDSS context rebuild was cancelled by close");
      if (created==registry.end() || registry.count(destination))
        throw std::runtime_error("cuDSS rebuild context adoption failed");
      auto factor=created->second;
      registry.emplace(destination,std::move(factor));
      registry.erase(created_id);
    }
    created_id=-1;
    return 0;
  } catch (const std::exception &e) {
    const std::string message=e.what();
    if (created_id>=0) kp_cudss_destroy(created_id);
    last_error=message;
    return -1;
  }
}
extern "C" int kp_cudss_rebuild_context(int64_t destination, int device,
    int64_t n, int64_t nnz, const void *rp, const void *ci, const void *av,
    int is_double, int spd, int64_t rhs_columns, const char *threading_lib) {
  return RebuildContext(destination,device,n,nnz,rp,ci,av,is_double,spd,rhs_columns,threading_lib,0);
}
extern "C" int kp_cudss_rebuild_context_with_reordering(int64_t destination, int device,
    int64_t n, int64_t nnz, const void *rp, const void *ci, const void *av,
    int is_double, int spd, int64_t rhs_columns, const char *threading_lib, int reordering) {
  return RebuildContext(destination,device,n,nnz,rp,ci,av,is_double,spd,rhs_columns,threading_lib,reordering);
}

template<ffi::DataType T> ffi::Error SolveImpl(cudaStream_t stream,
    ffi::Buffer<T> b, ffi::ResultBuffer<T> x, int64_t id,
    const void *new_values=nullptr, int64_t value_count=0, int64_t variant=-1, int64_t phase=0) {
  try {
    std::shared_ptr<Factor> f;
    { std::lock_guard<std::mutex> lock(registry_mutex);
      auto it=registry.find(id); if (it==registry.end()) return ffi::Error::InvalidArgument("cuDSS factor is closed");
      f=it->second; }
    std::lock_guard<std::mutex> lock(f->mutex); DeviceGuard guard(f->device);
    auto shape=b.dimensions();
    if (shape.size()!=2 || shape[0]!=f->n || shape[1]<1)
      return ffi::Error::InvalidArgument("cuDSS RHS shape must match the factor");
    if ((T==ffi::F64) != (f->dtype==CUDSS_R_64F))
      return ffi::Error::InvalidArgument("cuDSS factor and RHS dtypes differ");
    if (new_values && value_count!=f->nnz)
      return ffi::Error::InvalidArgument("cuDSS refactor values must match the prepared CSR pattern");
    cudaPointerAttributes attr{}; CheckCuda(cudaPointerGetAttributes(&attr,b.typed_data()));
    if (attr.device!=f->device) return ffi::Error::InvalidArgument("cuDSS factor and RHS devices differ");
    cudaStreamCaptureStatus capture;
    CheckCuda(cudaStreamIsCapturing(stream,&capture));
    if (capture!=cudaStreamCaptureStatusNone)
      return ffi::Error::InvalidArgument("cuDSS event-based solves do not support CUDA Graph capture");
    f->retire(false);
    Factor::Pending p;
    try {
      auto cached=std::find_if(f->available.begin(),f->available.end(),
                              [&](const Factor::Pending &v){return v.columns==shape[1];});
      if(cached!=f->available.end()) {
        p=*cached; f->available.erase(cached); ++f->descriptors_reused;
        Check(cudssMatrixSetValues(p.b,b.typed_data()));
        Check(cudssMatrixSetValues(p.x,x->typed_data()));
      } else {
        p.columns=shape[1]; ++f->descriptors_created;
        CheckCuda(cudaEventCreateWithFlags(&p.ready,cudaEventDisableTiming));
        CheckCuda(cudaEventCreateWithFlags(&p.done,cudaEventDisableTiming));
        Check(cudssMatrixCreateDn(&p.b,f->n,shape[1],f->n,b.typed_data(),f->dtype,CUDSS_LAYOUT_COL_MAJOR));
        Check(cudssMatrixCreateDn(&p.x,f->n,shape[1],f->n,x->typed_data(),f->dtype,CUDSS_LAYOUT_COL_MAJOR));
      }
      CheckCuda(cudaEventRecord(p.ready,stream));
      CheckCuda(cudaStreamWaitEvent(f->solve_stream,p.ready,0));
      if (new_values && (variant < 0 || variant != f->variant)) {
        // Previous solves, value update, refactor, and this solve are ordered
        // on the factor stream under one enqueue lock. No stale numerical
        // factors can leak across interleaved coefficient/adjoint calls.
        for(auto &entry:f->graphs) entry.second.clear();
        f->graphs.clear();
        CheckCuda(cudaMemcpyAsync(f->values,new_values,f->nnz*(T==ffi::F64 ? sizeof(double) : sizeof(float)),
                                  cudaMemcpyDeviceToDevice,f->solve_stream));
        Check(cudssExecute(f->handle,CUDSS_PHASE_REFACTORIZATION,f->config,f->data,f->a,p.x,p.b));
        ++f->refactorizations;
        f->variant = variant;
      }
      if(f->graph_enabled && !new_values && phase==0 && shape[1]<=f->rhs_columns) {
        // Fixed buffers belong to this factor. Copies and every replay share
        // solve_stream, so concurrent JAX callers cannot overwrite an RHS.
        size_t bytes=f->n*shape[1]*(T==ffi::F64 ? sizeof(double) : sizeof(float));
        auto &g=f->graphs[shape[1]];
        CheckCuda(cudaMemcpyAsync(f->rhs_storage,b.typed_data(),bytes,cudaMemcpyDeviceToDevice,f->solve_stream));
        if(!g.executable) {
          g.clear();
          try {
          if(!g.b) Check(cudssMatrixCreateDn(&g.b,f->n,shape[1],f->n,f->rhs_storage,f->dtype,CUDSS_LAYOUT_COL_MAJOR));
          if(!g.x) Check(cudssMatrixCreateDn(&g.x,f->n,shape[1],f->n,f->solution_storage,f->dtype,CUDSS_LAYOUT_COL_MAJOR));
          Check(cudssExecute(f->handle,CUDSS_PHASE_SOLVE,f->config,f->data,f->a,g.x,g.b));
          CheckCuda(cudaStreamSynchronize(f->solve_stream));
          CheckCuda(cudaStreamBeginCapture(f->solve_stream,cudaStreamCaptureModeThreadLocal));
          auto status=cudssExecute(f->handle,CUDSS_PHASE_SOLVE,f->config,f->data,f->a,g.x,g.b);
          auto ended=cudaStreamEndCapture(f->solve_stream,&g.graph);
          Check(status); CheckCuda(ended);
          CheckCuda(cudaGraphInstantiate(&g.executable,g.graph,0));
          } catch (...) {
            // A failed capture/instantiate must not leave handles to be
            // overwritten by the next attempt at this RHS width.
            cudaStreamSynchronize(f->solve_stream);
            g.clear();
            throw;
          }
        }
        CheckCuda(cudaGraphLaunch(g.executable,f->solve_stream)); ++f->graph_launches;
        CheckCuda(cudaMemcpyAsync(x->typed_data(),f->solution_storage,bytes,cudaMemcpyDeviceToDevice,f->solve_stream));
      } else {
        Check(cudssExecute(f->handle,phase ? static_cast<cudssPhase_t>(phase) : CUDSS_PHASE_SOLVE,f->config,f->data,f->a,p.x,p.b));
      }
      CheckCuda(cudaEventRecord(p.done,f->solve_stream));
      CheckCuda(cudaStreamWaitEvent(stream,p.done,0));
      f->pending.push_back(p); ++f->solves;
    } catch (...) {
      f->variant = -1;
      cudaStreamSynchronize(f->solve_stream);
      if(p.b) cudssMatrixDestroy(p.b); if(p.x) cudssMatrixDestroy(p.x);
      if(p.ready) cudaEventDestroy(p.ready); if(p.done) cudaEventDestroy(p.done);
      throw;
    }
    return ffi::Error::Success();
  } catch (const std::exception &e) { return ffi::Error::Internal(e.what()); }
}
template<ffi::DataType T> ffi::Error Solve(cudaStream_t stream,
    ffi::Buffer<T> b, ffi::ResultBuffer<T> x, int64_t id) {
  return SolveImpl<T>(stream,b,x,id);
}
template<ffi::DataType T> ffi::Error RefactorSolve(cudaStream_t stream,
    ffi::Buffer<T> values, ffi::Buffer<T> b, ffi::ResultBuffer<T> x, int64_t id) {
  if(values.dimensions().size()!=1)
    return ffi::Error::InvalidArgument("cuDSS refactor values must be a vector");
  return SolveImpl<T>(stream,b,x,id,values.typed_data(),values.dimensions()[0]);
}
template<ffi::DataType T> ffi::Error VariantSolve(cudaStream_t stream,
    ffi::Buffer<T> values, ffi::Buffer<T> b, ffi::ResultBuffer<T> x,
    int64_t id, int64_t variant) {
  if(values.dimensions().size()!=1 || variant < 0)
    return ffi::Error::InvalidArgument("cuDSS frozen variant requires vector values and a nonnegative key");
  return SolveImpl<T>(stream,b,x,id,values.typed_data(),values.dimensions()[0],variant);
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_variant_f64, VariantSolve<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int64_t>("factor_id").Attr<int64_t>("variant_id"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_variant_f32, VariantSolve<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>().Attr<int64_t>("factor_id").Attr<int64_t>("variant_id"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_refactor_f64, RefactorSolve<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int64_t>("factor_id"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_refactor_f32, RefactorSolve<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>().Attr<int64_t>("factor_id"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_solve_f64, Solve<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int64_t>("factor_id"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_solve_f32, Solve<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>().Attr<int64_t>("factor_id"));

// Changing-matrix path, used for coefficient inverse problems. The native
// operation owns transient factors; JAX supplies the implicit AD rule.
// Canonicalization is host-side; numerical factorization/solve run on CUDA.
template<ffi::DataType T> ffi::Error SolveCoo(cudaStream_t stream,
    ffi::Buffer<ffi::S32> indices, ffi::Buffer<T> values,
    ffi::Buffer<T> rhs, ffi::ResultBuffer<T> solution, bool transpose) {
  using Scalar = std::conditional_t<T==ffi::F64, double, float>;
  int64_t id = -1;
  void *dr = nullptr, *dc = nullptr, *dv = nullptr;
  try {
    auto is = indices.dimensions(); auto bs = rhs.dimensions();
    if (is.size()!=2 || is[1]!=2 || values.dimensions().size()!=1 ||
        values.dimensions()[0]!=is[0] || bs.size()!=2 || bs[0]<1 || bs[1]<1)
      return ffi::Error::InvalidArgument("invalid cuDSS COO/RHS dimensions");
    const int64_t n = bs[0], nnz = is[0];
    std::vector<int32_t> ij(2*nnz); std::vector<Scalar> av(nnz);
    CheckCuda(cudaMemcpyAsync(ij.data(), indices.typed_data(), ij.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaMemcpyAsync(av.data(), values.typed_data(), av.size()*sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    std::vector<std::tuple<int32_t,int32_t,Scalar>> entries;
    entries.reserve(nnz);
    for (int64_t k=0;k<nnz;++k) {
      int32_t r=ij[2*k], c=ij[2*k+1];
      if (transpose) std::swap(r,c);
      if (r<0 || c<0 || r>=n || c>=n) throw std::runtime_error("COO index out of range");
      entries.emplace_back(r,c,av[k]);
    }
    std::sort(entries.begin(), entries.end(), [](const auto &a,const auto &b) {
      return std::tie(std::get<0>(a),std::get<1>(a)) < std::tie(std::get<0>(b),std::get<1>(b));
    });
    std::vector<int32_t> rp(n+1,0), ci; std::vector<Scalar> data;
    for (const auto &e:entries) {
      int32_t r=std::get<0>(e), c=std::get<1>(e);
      if (!ci.empty() && rp[r+1]>0 && ci.back()==c) data.back()+=std::get<2>(e);
      else { ci.push_back(c); data.push_back(std::get<2>(e)); ++rp[r+1]; }
    }
    for (int64_t r=0;r<n;++r) rp[r+1]+=rp[r];
    CheckCuda(cudaMalloc(&dr,rp.size()*sizeof(int32_t)));
    CheckCuda(cudaMalloc(&dc,ci.size()*sizeof(int32_t)));
    CheckCuda(cudaMalloc(&dv,data.size()*sizeof(Scalar)));
    CheckCuda(cudaMemcpy(dr,rp.data(),rp.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dc,ci.data(),ci.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(dv,data.data(),data.size()*sizeof(Scalar),cudaMemcpyHostToDevice));
    cudaPointerAttributes attr{}; CheckCuda(cudaPointerGetAttributes(&attr,rhs.typed_data()));
    id=kp_cudss_create(attr.device,n,ci.size(),dr,dc,dv,T==ffi::F64,0,bs[1],"");
    if (id<0) throw std::runtime_error(last_error);
    auto result=Solve<T>(stream,rhs,solution,id);
    if (kp_cudss_check_status(id)<0) throw std::runtime_error(last_error);
    kp_cudss_destroy(id); cudaFree(dr); cudaFree(dc); cudaFree(dv);
    return result;
  } catch (const std::exception &e) {
    if (id>=0) kp_cudss_destroy(id);
    cudaFree(dr); cudaFree(dc); cudaFree(dv);
    return ffi::Error::Internal(e.what());
  }
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_coo_f64, SolveCoo<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>().Attr<bool>("transpose"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_coo_f32, SolveCoo<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>().Attr<bool>("transpose"));

template<ffi::DataType T> ffi::Error SolveCsr(cudaStream_t stream,
    ffi::Buffer<ffi::S32> rows, ffi::Buffer<ffi::S32> cols,
    ffi::Buffer<T> values, ffi::Buffer<T> rhs, ffi::ResultBuffer<T> solution) {
  int64_t id=-1;
  try {
    auto bs=rhs.dimensions();
    if(bs.size()!=2 || bs[0]<1 || bs[1]<1 || rows.dimensions().size()!=1 ||
       rows.dimensions()[0]!=bs[0]+1 || cols.dimensions().size()!=1 ||
       values.dimensions().size()!=1 || cols.dimensions()[0]!=values.dimensions()[0])
      return ffi::Error::InvalidArgument("invalid cuDSS CSR/RHS dimensions");
    // Setup/factorization still needs completed input values; there is no
    // matrix download or CPU sorting in this prepared-pattern path.
    CheckCuda(cudaStreamSynchronize(stream));
    cudaPointerAttributes attr{}; CheckCuda(cudaPointerGetAttributes(&attr,rhs.typed_data()));
    id=kp_cudss_create(attr.device,bs[0],values.dimensions()[0],rows.typed_data(),
                      cols.typed_data(),values.typed_data(),T==ffi::F64,0,bs[1],"");
    if(id<0) throw std::runtime_error(last_error);
    auto result=Solve<T>(stream,rhs,solution,id);
    if(kp_cudss_check_status(id)<0) throw std::runtime_error(last_error);
    kp_cudss_destroy(id); return result;
  } catch(const std::exception &e) {
    if(id>=0) kp_cudss_destroy(id);
    return ffi::Error::Internal(e.what());
  }
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_csr_f64, SolveCsr<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
    .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_csr_f32, SolveCsr<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
    .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>());

// SPD-only factor actions. Forward includes the fill-reducing permutation;
// backward applies its inverse, so their composition is A^{-1}.
template<ffi::DataType T> ffi::Error TriangularSolve(cudaStream_t stream,
    ffi::Buffer<T> b, ffi::ResultBuffer<T> x, int64_t id, int64_t backward) {
  auto phase = backward ? (CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM)
                        : (CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD);
  return SolveImpl<T>(stream,b,x,id,nullptr,0,-1,phase);
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_triangular_f64, TriangularSolve<ffi::F64>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F64>>()
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int64_t>("factor_id").Attr<int64_t>("backward"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_triangular_f32, TriangularSolve<ffi::F32>,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>()
    .Ret<ffi::Buffer<ffi::F32>>().Attr<int64_t>("factor_id").Attr<int64_t>("backward"));

// Export the reordering directly to a GPU output. It lets bounded permuted
// identity blocks use the existing checked SPD forward action; we need no
// undocumented raw-forward work-buffer semantics or host permutation copy.
ffi::Error ReorderingPermutation(cudaStream_t stream,
    ffi::ResultBuffer<ffi::S32> result, int64_t id) {
  cudaEvent_t ready=nullptr, done=nullptr;
  try {
    std::shared_ptr<Factor> f;
    { std::lock_guard<std::mutex> lock(registry_mutex);
      auto it=registry.find(id); if(it==registry.end())
        return ffi::Error::InvalidArgument("cuDSS factor is closed");
      f=it->second; }
    std::lock_guard<std::mutex> lock(f->mutex); DeviceGuard guard(f->device);
    auto shape=result->dimensions();
    if(shape.size()!=1 || shape[0]!=f->n)
      return ffi::Error::InvalidArgument("cuDSS permutation output must have N entries");
    CheckCuda(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    CheckCuda(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    CheckCuda(cudaEventRecord(ready,stream));
    CheckCuda(cudaStreamWaitEvent(f->solve_stream,ready,0));
    size_t written=0;
    Check(cudssDataGet(f->handle,f->data,CUDSS_DATA_PERM_REORDER_ROW,
      result->typed_data(),f->n*sizeof(int32_t),&written));
    if(written!=f->n*sizeof(int32_t))
      throw std::runtime_error("cuDSS returned an incomplete reordering permutation");
    CheckCuda(cudaEventRecord(done,f->solve_stream));
    CheckCuda(cudaStreamWaitEvent(stream,done,0));
    CheckCuda(cudaEventDestroy(ready)); ready=nullptr;
    CheckCuda(cudaEventDestroy(done)); done=nullptr;
    return ffi::Error::Success();
  } catch(const std::exception &e) {
    if(ready) cudaEventDestroy(ready);
    if(done) cudaEventDestroy(done);
    return ffi::Error::Internal(e.what());
  }
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(kp_cudss_permutation, ReorderingPermutation,
  ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Ret<ffi::Buffer<ffi::S32>>().Attr<int64_t>("factor_id"));

extern "C" int kp_cudss_resource_counts(int64_t id, int64_t *created, int64_t *reused) {
  std::shared_ptr<Factor> f;
  {std::lock_guard<std::mutex> lock(registry_mutex);
   auto it=registry.find(id); if(it==registry.end()) return -1; f=it->second;}
  std::lock_guard<std::mutex> lock(f->mutex);
  *created=f->descriptors_created; *reused=f->descriptors_reused; return 0;
}

// cuDSS metadata are scalar planning/storage diagnostics, never numerical
// matrix values. Expose them so paper timings cannot mistake input CSR nnz
// for Cholesky fill or total permanent/peak factor workspace.
extern "C" int kp_cudss_factor_stats(int64_t id, int64_t *factor_nnz,
    int64_t *memory_estimates) {
  try {
    std::shared_ptr<Factor> f;
    { std::lock_guard<std::mutex> lock(registry_mutex);
      auto it=registry.find(id); if(it==registry.end())
        throw std::runtime_error("cuDSS factor is closed");
      f=it->second; }
    std::lock_guard<std::mutex> lock(f->mutex); DeviceGuard guard(f->device);
    size_t written=0;
    Check(cudssDataGet(f->handle,f->data,CUDSS_DATA_LU_NNZ,
      factor_nnz,sizeof(int64_t),&written));
    Check(cudssDataGet(f->handle,f->data,CUDSS_DATA_MEMORY_ESTIMATES,
      memory_estimates,16*sizeof(int64_t),&written));
    return 0;
  } catch(const std::exception &e) { last_error=e.what(); return -1; }
}

// Host controls are serialized with all enqueue operations. Graph execution
// remains internal; this does not advertise whole-XLA-graph compatibility.
extern "C" int64_t kp_cudss_graph_mode(int64_t id, int enabled) {
  std::shared_ptr<Factor> f;
  { std::lock_guard<std::mutex> lock(registry_mutex);
    auto it=registry.find(id); if(it==registry.end()) return -1; f=it->second; }
  std::lock_guard<std::mutex> lock(f->mutex);
  if(enabled>=0) f->graph_enabled=enabled!=0;
  return f->graph_launches;
}
