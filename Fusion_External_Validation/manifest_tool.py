"""Hash regular bundle files; runtime outputs belong outside this directory."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inventory(root=ROOT):
    return {str(p.relative_to(root)): {
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size
    } for p in sorted(root.rglob('*')) if p.is_file()
        and p != root / 'MANIFEST.json'
        and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def verify(root=ROOT):
    stored = json.loads((root / 'MANIFEST.json').read_text())['files']
    actual = inventory(root)
    missing = sorted(set(stored) - set(actual))
    unexpected = sorted(set(actual) - set(stored))
    changed = sorted(k for k in set(stored) & set(actual) if stored[k] != actual[k])
    return dict(passed=not (missing or unexpected or changed), files=len(stored),
                missing=missing, unexpected=unexpected, changed=changed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', help='Refresh after adding deliverables; never a historical experiment freeze.')
    args = p.parse_args()
    if args.write:
        files = inventory()
        out = dict(schema_version=1, kind='Delivery integrity manifest, excluding itself and bytecode',
                   historical_freezes='Preserved separately in original protocol and freeze files',
                   files=files, total_bytes=sum(v['bytes'] for v in files.values()))
        (ROOT / 'MANIFEST.json').write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(dict(files=len(files), bytes=out['total_bytes'])))
    else:
        out = verify()
        print(json.dumps(out, indent=2))
        if not out['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
