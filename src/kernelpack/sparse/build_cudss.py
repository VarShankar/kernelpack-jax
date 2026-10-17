"""Explicit optional build: python -m kernelpack.sparse.build_cudss --help."""

from pathlib import Path
import argparse, os, subprocess, sys
import jax


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cudss-root", default=os.environ.get("CUDSS_ROOT"))
    p.add_argument("--cuda-root", default=os.environ.get("CUDA_ROOT"))
    p.add_argument("--output")
    args = p.parse_args()
    package = Path(__file__).parent
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    cuda = Path(args.cuda_root) if args.cuda_root else site / "nvidia/cu13"
    cudss = Path(args.cudss_root) if args.cudss_root else site / "nvidia/cu13"
    cudss_lib = next(iter((cudss / "lib").glob("libcudss.so*")), None)
    runtime_lib = next(iter((cuda / "lib").glob("libcudart.so*")), None)
    if (
        not (cudss / "include/cudss.h").exists()
        or not (cuda / "include/cuda_runtime.h").exists()
        or not cudss_lib
        or not runtime_lib
    ):
        p.error(
            "Supply CUDA/cuDSS roots containing include/ and lib/; see docs/fastol_gpu.md"
        )
    out = Path(args.output) if args.output else package / "_cudss_ffi.so"
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-pthread",
        str(package / "_native/cudss_ffi.cc"),
        "-I" + jax.ffi.include_dir(),
        "-I" + str(cudss / "include"),
        "-I" + str(cuda / "include"),
        str(cudss_lib),
        str(runtime_lib),
        "-Wl,-rpath," + str(cudss / "lib"),
        "-Wl,-rpath," + str(cuda / "lib"),
        "-o",
        str(out),
    ]
    subprocess.run(command, check=True)
    print(out)


if __name__ == "__main__":
    main()
