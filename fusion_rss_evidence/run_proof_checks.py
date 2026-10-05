"""Run unchanged synthetic proof checks in a new output directory."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Proof output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    source = root / "theory/check_retained_pairing_bridge.py"
    copied = output / source.name
    shutil.copyfile(source, copied)
    assert source.read_bytes() == copied.read_bytes()
    result = subprocess.run([sys.executable, str(copied)], capture_output=True, text=True)
    (output / "execution.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    report = {
        "passed": True,
        "scope": "Synthetic posterior-bridge identities and perturbation bounds; not physical outcomes",
        "unchanged_check_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "summary": json.loads(result.stdout),
    }
    (output / "execution.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
