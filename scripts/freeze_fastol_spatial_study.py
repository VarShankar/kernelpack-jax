"""Copy and hash the spatial study's Python sources without executing numerics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all",
        "--", "src", "scripts", "tests"], cwd=root, text=True)
    if status.strip():
        raise RuntimeError("Commit the reviewed production/study sources before freezing:\n" + status)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source_suffixes = {".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".pyd", ".so"}
    paths = sorted(p for p in (root / "src/kernelpack").rglob("*")
                   if p.is_file() and p.suffix.lower() in source_suffixes)
    paths += [root / "scripts" / name for name in (
        "fastol_spatial_study_core.py", "fastol_spatial_properties.py",
        "fastol_spatial_statistics.py", "fastol_scattered_clouds.py",
        "run_fastol_spatial_validation.py",
        "report_fastol_spatial_properties.py", "freeze_fastol_spatial_study.py")]
    paths += [root / "scripts/matlab/plot_fastol_spatial_properties.m"]
    paths += sorted(p for p in (root / "tests").rglob("*.py") if p.is_file())
    pins = {}
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        pins[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    for relative, digest in pins.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed while freezing: {relative}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--", "src", "scripts", "tests"], cwd=root)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), origin=str(root),
                    commit=head, tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                    source_sha256=pins, numerical_execution=False)
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(destination=str(destination), files=len(pins), commit=head)))


if __name__ == "__main__":
    main()
