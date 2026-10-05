"""Constructed header-only parser checks; never opens the official archive."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHYSICAL = HERE.parent/'physical'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def text_fixture(header):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    start = dt.datetime(2026, 1, 1)
    for row in range(151):
        stamp = start+dt.timedelta(milliseconds=20*row, seconds=2*(row >= 81))
        scientific = dict(timestamp=stamp.isoformat(sep=' '), back_x=row/10., back_y=-row/20., back_z=1.,
            thigh_x=row/40., thigh_y=2., thigh_z=-.1, label=1+(row % 2))
        writer.writerow([scientific[name] if name in scientific else f'inert-export-{row}' for name in header])
    stream.seek(0)
    return stream


def main():
    api = module('amended_header_parser', PHYSICAL/'adapter.py')
    old = module('preserved_header_parser', PHYSICAL/'freeze_v2_before_header_amendment/adapter.py')
    reference, reference_audit = old.parse_recording(text_fixture(old.FIELDS), 'synthetic')
    valid = [api.FIELDS, ('index',)+api.FIELDS, (api.FIELDS[0], 'index')+api.FIELDS[1:],
             ('',)+api.FIELDS, api.FIELDS+('',)]
    for header in valid:
        rows, audit = api.parse_recording(text_fixture(header), 'synthetic')
        assert len(rows) == len(reference) == 4
        for row, expected in zip(rows, reference):
            assert set(row) == set(expected)
            for key in row:
                if isinstance(row[key], np.ndarray):
                    assert np.array_equal(row[key], expected[key])
                else:
                    assert row[key] == expected[key]
        for key, expected in reference_audit.items():
            assert audit[key] == expected
        assert audit['header_fields'] == list(header)
        assert audit['header_field_count'] == len(header)
        assert audit['scientific_header_fields'] == list(api.FIELDS)
        assert audit['ignored_export_columns'] == [name for name in header if name not in api.FIELDS]
    invalid = [api.FIELDS+('other',), api.FIELDS+('label',), api.FIELDS+('index', ''),
               ('back_x', 'timestamp')+api.FIELDS[2:], api.FIELDS[:-1], api.FIELDS+('index', 'index')]
    for header in invalid:
        try:
            api.parse_recording(text_fixture(header), 'synthetic')
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid header admitted', header))
    for values in (['2026-01-01 00:00:00', 0, 0, 0, 0, 0, 0],
                   ['2026-01-01 00:00:00', 0, 0, 0, 0, 0, 0, 1, 'unheaded-extra']):
        stream = io.StringIO(); writer = csv.writer(stream)
        writer.writerow(api.FIELDS); writer.writerow(values); stream.seek(0)
        try:
            api.parse_recording(stream, 'synthetic')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid row width admitted')
    report = dict(passed=True, scope='synthetic data only; no official data opened', valid_headers=len(valid),
        invalid_header_forms_rejected=len(invalid), invalid_row_width_forms_rejected=2,
        eight_and_nine_column_retained_scientific_arrays_identical=True,
        preserved_original_eight_column_parser_identical=True, retained_indices=[row['raw_row'] for row in reference],
        gap_history_runs=reference_audit['history_runs'], rows_labels_values_roles_features_unchanged=True,
        adapter_sha256_after=hashlib.sha256((PHYSICAL/'adapter.py').read_bytes()).hexdigest())
    path = HERE/'header_amendment_report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    amendment_path = PHYSICAL/'SCHEMA_HEADER_AMENDMENT.json'
    amendment = json.loads(amendment_path.read_text())
    amendment.update(status='parser_patched_and_synthetic_checks_passed_awaiting_root_hash_rebinding',
        adapter_sha256_after=report['adapter_sha256_after'], synthetic_validation_report=str(path.resolve()),
        header_rule='exact eight scientific fields in original order; at most one inert index or empty-name export column; all other/repeated fields and inconsistent row widths rejected',
        metadata_audit_fields=['header_fields', 'header_field_count', 'scientific_header_fields', 'ignored_export_columns'])
    amendment_path.write_text(json.dumps(amendment, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
