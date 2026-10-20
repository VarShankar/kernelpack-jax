"""Run the spatial regression proof on CUDA64 and authenticate its exact sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


SUITES = (
    "tests/test_divfree_center_limits.py",
    "tests/test_jax_spatial_poisson_periodic.py",
    "tests/test_geometry_device_pipeline.py",
    "tests/test_cfpu_device_build.py",
    "tests/test_domain_stencil_cache_invalidation.py",
)


def hash_sources(root, expected):
    actual = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in expected}
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError("Frozen validation sources changed: "+", ".join(changed))
    return actual


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_manifest = root/"source_manifest.json"
    expected = json.loads(source_manifest.read_text())["source_sha256"]
    for path in (args.junit, args.manifest):
        if path.exists():
            raise FileExistsError(path)
        if path.resolve().is_relative_to(root):
            raise ValueError("Validation artifacts must be outside the frozen source directory")
        path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path[:0] = [str(root/"src"), str(root/"scripts")]
    before = hash_sources(root, expected)
    started = time.perf_counter()
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        source_before_sha256=before, selected_test_suites=SUITES,
        platform="cuda", dtype="float64", returncode=None)
    returncode = 1
    try:
        import jax
        from jax.extend import backend
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        device = jax.devices("gpu")[0]
        if device.platform != "gpu" or "nvidia" not in device.device_kind.lower():
            raise RuntimeError(f"NVIDIA CUDA is required: {device}")
        jax.config.update("jax_default_device", device)
        import kernelpack
        if not Path(kernelpack.__file__).resolve().is_relative_to(root/"src"):
            raise RuntimeError("Validation imported kernelpack outside the frozen source")
        from kernelpack.accelerators.jax_spatial import enable_compilation_cache
        report.update(device=str(device), device_kind=device.device_kind,
            backend_platform_version=backend.get_backend().platform_version,
            jax_version=importlib.metadata.version("jax"),
            compilation_cache=enable_compilation_cache(),
            imported_kernelpack=str(Path(kernelpack.__file__).resolve()))
        import pytest
        returncode = int(pytest.main(["-q", "-rfE", "--tb=short",
            "--junitxml="+str(args.junit.resolve()),
            *(str(root/name) for name in SUITES)]))
        if not jax.config.x64_enabled or jax.devices()[0].platform != "gpu":
            raise RuntimeError("Validation changed the required CUDA64 configuration")
    except BaseException:
        report["traceback"] = traceback.format_exc()
        returncode = 1
    finally:
        report["returncode"] = returncode
        report["elapsed_seconds"] = time.perf_counter()-started
        try:
            report["source_after_sha256"] = hash_sources(root, expected)
        except Exception:
            report["source_check_traceback"] = traceback.format_exc()
            returncode = report["returncode"] = 1
        if args.junit.exists():
            report["junit_sha256"] = hashlib.sha256(args.junit.read_bytes()).hexdigest()
        with args.manifest.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(dict(validation_returncode=returncode,
            elapsed_seconds=report["elapsed_seconds"], manifest=str(args.manifest))), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
