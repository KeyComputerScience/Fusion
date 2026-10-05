"""Build and SHA-verify a standalone ZIP from an already verified delivery manifest."""
from __future__ import annotations
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
sys.dont_write_bytecode = True
from manifest_tool import verify

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT.with_suffix('.zip'))
    args = p.parse_args(); output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit('Archive must be outside the delivered bundle')
    report = verify()
    if not report['passed']:
        raise SystemExit('Manifest mismatch; inspect before making a release: ' + json.dumps(report))
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    files = [*sorted(manifest['files']), 'MANIFEST.json']
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in files:
            archive.write(ROOT / relative, ROOT.name + '/' + relative)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise SystemExit('ZIP CRC verification failed')
        for relative, item in manifest['files'].items():
            content = archive.read(ROOT.name + '/' + relative)
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise SystemExit('ZIP content checksum failed: ' + relative)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '  ' + output.name + '\n')
    result = dict(passed=True, archive=str(output), archive_sha256=digest,
        archive_bytes=output.stat().st_size, archived_files=len(files),
        uncompressed_bytes=sum(item['bytes'] for item in manifest['files'].values()),
        manifest_and_all_archived_sha256_verified=True)
    output.with_suffix(output.suffix + '.verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
