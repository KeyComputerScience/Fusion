"""Transport integrity only: never parse sensor values or execute mirror code."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys, zipfile, zlib

ROOT = Path(__file__).resolve().parent
COMMIT = 'e90bded40b26e32edb92f6ff6e1c255aebb3ab37'
PREFIX = 'har-with-opportunity-dataset-' + COMMIT + '/'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def verify(source):
    tree = json.loads((ROOT / 'mirror_tree_metadata.json').read_text())
    assert tree['sha'] == COMMIT and not tree['truncated']
    expected = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
    independent = json.loads((ROOT.parent / 'risk/opportunity_mirror_verified_metadata.json').read_text())
    assert independent['commit_sha'] == COMMIT and not independent['tree_truncated']
    independent_files = {x['path']: x for x in independent['files'] + independent['metadata_files']}
    known = json.loads((ROOT / 'mirror_official_binary_provenance.json').read_text())
    official = {x['name']: x for x in known['records'] if x.get('complete')}
    required = [f'OpportunityUCIDataset/dataset/S{p}-{s}.dat'
                for p in range(1, 5) for s in ('Drill', 'ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5')]
    required += ['OpportunityUCIDataset/dataset/column_names.txt',
                 'OpportunityUCIDataset/dataset/label_legend.txt']
    rows = []
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        assert all(PREFIX + n in names for n in required), 'Incomplete frozen collection'
        for name in required:
            info = archive.getinfo(PREFIX + name)
            want = expected[name]
            assert want['sha'] == independent_files[name]['sha'] and want['size'] == independent_files[name]['size']
            assert info.file_size == want['size'], (name, info.file_size, want['size'])
            h = hashlib.sha1(b'blob ' + str(info.file_size).encode() + b'\0')
            h256 = hashlib.sha256(); crc = 0; count = 0
            with archive.open(info) as f:
                for b in iter(lambda: f.read(1 << 20), b''):
                    h.update(b); h256.update(b); crc = zlib.crc32(b, crc); count += len(b)
            assert count == info.file_size and crc == info.CRC, ('CRC/length', name)
            assert h.hexdigest() == want['sha'], ('Pinned Git blob', name)
            if name in official:
                old = official[name]
                assert old['sha256'] == h256.hexdigest(), ('Official byte equality', name)
                assert old['official_header_crc32'] == crc, ('Official CRC equality', name)
            rows.append(dict(name=name, bytes=count, crc32=crc, git_blob_sha1=h.hexdigest(),
                             sha256=h256.hexdigest(), matches_known_official=name in official))
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pinned_commit=COMMIT, source_path=str(Path(source).resolve()),
                  independently_resolved_root_tree=independent['tree_sha'],
                  source_sha256=sha(source), source_bytes=Path(source).stat().st_size,
                  all24_native_required=True, all26_required_crc_and_git_hashes_valid=True,
                  known_official_dat_byte_matches=sum(r['matches_known_official'] and r['name'].endswith('.dat') for r in rows),
                  both_official_legends_match=True, sensor_or_label_numeric_decoding=False,
                  code_from_mirror_executed=False, required_members=rows)
    (ROOT / 'opportunity_mirror_integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'required_members'}, indent=2), flush=True)
    return report

def repack(source):
    report = verify(source)
    destination = ROOT / 'raw/opportunity226.zip'
    assert not destination.exists(), 'Do not overwrite an existing collection'
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as derived:
        for record in report['required_members']:
            name = record['name']
            with original.open(PREFIX + name) as src, derived.open(name, 'w', force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 26
    report.update(derived_transport_zip=str(destination.resolve()), derived_sha256=sha(destination),
                  derived_bytes=destination.stat().st_size, derived_original_names=True,
                  origin='Pinned third-party mirror; 15 native members and both legends identical to official UCI bytes, remaining 9 pinned-mirror provenance',
                  official_archive_sha_claim=False)
    (ROOT / 'opportunity_acquisition_completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DERIVED_TRANSPORT_ARCHIVE', destination, report['derived_sha256'], flush=True)

if __name__ == '__main__':
    operation, source = sys.argv[1:]
    {'verify': verify, 'repack': repack}[operation](Path(source))
