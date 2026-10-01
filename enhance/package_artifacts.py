"""Build a self-contained archive with source and artifact checksums."""
import argparse
import json
from pathlib import Path
import zipfile

from data_pipeline import ROOT, sha256, write_json


def included(path):
    relative = path.relative_to(ROOT)
    return (path.is_file() and not any(p in ("__pycache__", "smoke_data", "smoke_results") for p in relative.parts)
            and path.suffix not in (".pyc", ".part") and path.name != "artifact_hashes.json"
            and not any(p.startswith(".") for p in relative.parts))


def package(destination):
    paths = sorted(p for p in ROOT.rglob("*") if included(p) and p.resolve() != destination.resolve())
    write_json(ROOT / "artifact_hashes.json", {
        "algorithm": "sha256", "archive_root": ROOT.name,
        "files": [{"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]})
    paths.append(ROOT / "artifact_hashes.json")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as archive:
        problem = archive.testzip()
        if problem:
            raise ValueError(f"Archive CRC failed: {problem}")
    print(json.dumps({"path": str(destination), "files": len(paths), "bytes": destination.stat().st_size,
                      "sha256": sha256(destination), "crc_verified": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "fusion_performance_reproducible.zip")
    package(parser.parse_args().output)
