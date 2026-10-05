"""Metadata-only synthetic checks of HARTH adapter chronology and prefix fit.

No archive or physical cache is opened and no acquisition is authorized.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PATH = HERE.parent/'physical/adapter.py'
sp = importlib.util.spec_from_file_location('theory_fresh_adapter',PATH)
adapter = importlib.util.module_from_spec(sp)
sp.loader.exec_module(adapter)


class PrefixEngine:
    """Read-only stand-in: record fit arrays and return fixed zero models."""
    def __init__(self):
        self.calls = []

    def gradient(self, initial, x, y, steps, cfg, classes):
        self.calls.append((x.copy(),y.copy()))
        return initial.copy()

    @staticmethod
    def softmax(logits):
        z = np.exp(logits-logits.max(axis=1,keepdims=True))
        return z/z.sum(axis=1,keepdims=True)

    @staticmethod
    def context(x,groups):
        return np.array([np.mean(x[:,np.array(g)*3]) for g in groups])


def parser_check():
    def body(label):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(adapter.FIELDS)
        first = datetime.datetime(2026,1,1)
        for i in range(151):
            elapsed = .02*i+(2 if i>=60 else 0)
            stamp = first+datetime.timedelta(seconds=elapsed)
            writer.writerow([stamp.isoformat()]+[j+.001*i for j in range(6)]+[label])
        buffer.seek(0)
        return buffer
    kept,summary = adapter.parse_recording(body(1),'S1')
    poison,_ = adapter.parse_recording(body(2),'S1')
    assert [x['raw_row'] for x in kept] == [0,50,100,150]
    assert [x['session'] for x in kept] == ['S1/run0001']*2+['S1/run0002']*2
    assert summary['raw_timestamp_gaps_above_one_second'] == 1
    assert summary['history_runs'] == 2
    for a,b in zip(kept,poison):
        assert np.array_equal(a['raw'],b['raw'])
        assert (a['raw_row'],a['session'],a['origin']) == (b['raw_row'],b['session'],b['origin'])
    raw = np.arange(24.).reshape(4,6)
    x = adapter.boundary_lagged(raw,np.zeros(6),np.ones(6)*10,np.array(['a','a','b','b']))
    expected = np.stack([np.stack([raw[0],raw[0],raw[0]],axis=1),
                         np.stack([raw[1],raw[0],raw[0]],axis=1),
                         np.stack([raw[2],raw[2],raw[2]],axis=1),
                         np.stack([raw[3],raw[2],raw[2]],axis=1)])/10
    error = float(np.max(abs(x[:,:18]-expected.reshape(4,18))))
    assert error == 0 and np.array_equal(x[:,-1],np.ones(4))
    changed = raw.copy()
    changed[1:] += 100000
    changed_x = adapter.boundary_lagged(changed,np.zeros(6),np.ones(6)*10,np.array(['a','a','b','b']))
    assert np.array_equal(x[0],changed_x[0])
    return dict(passed=True,retained_original_rows=[x['raw_row'] for x in kept],
                raw_gap_resets_history=True,label_poison_does_not_change_selection_or_provenance=True,
                current_plus_previous_two_retained_rows=True,lag_reset_max_error=error,
                future_covariate_poison_leaves_first_issued_feature_unchanged=True)


def prefix_check():
    roster = [f'S{i}' for i in range(1,23)]
    n = 2200
    recording = np.repeat(roster,100)
    data = dict(raw=np.sin(np.arange(n*6).reshape(n,6)/127),y=np.arange(n)%12,
                recording=recording,session=np.array([r+'/run0001' for r in recording]),
                timestamp=[f'{recording[i]}/raw{i:08d}' for i in range(n)],
                metadata=dict(split_boundaries=dict(ntrain=800,ncal=1200),
                              fit_participants=roster[:8],calibration_participants=roster[8:12],
                              test_participants=roster[12:],participants=roster))
    cfg = dict(initial_steps=1,window=32)
    first_engine = PrefixEngine()
    pre = adapter.make_prefix(data,cfg,first_engine)
    changed = dict(data,raw=data['raw'].copy(),y=data['y'].copy())
    changed['raw'][800:] += 1e8
    changed['y'][800:] = 11-changed['y'][800:]
    second_engine = PrefixEngine()
    poisoned = adapter.make_prefix(changed,cfg,second_engine)
    assert pre['split']['training_rows'] == 8*70
    assert pre['split']['audit_rows'] == 8*30
    assert np.array_equal(pre['mean'],poisoned['mean']) and np.array_equal(pre['scale'],poisoned['scale'])
    assert np.array_equal(pre['q'],poisoned['q'])
    assert len(first_engine.calls) == len(second_engine.calls) == 3
    for first,second in zip(first_engine.calls,second_engine.calls):
        assert np.array_equal(first[0],second[0]) and np.array_equal(first[1],second[1])
    assert len(pre['prior']) == len(poisoned['prior']) == 8
    for first,second in zip(pre['prior'],poisoned['prior']):
        assert np.array_equal(first['x'],second['x']) and np.array_equal(first['y'],second['y'])
    assert [len(x) for x in pre['features']] == [10,10,19]
    assert len(pre['recording_streams']) == 14
    assert sum(x['partition']=='calibration' for x in pre['recording_streams']) == 4
    assert sum(x['partition']=='test' for x in pre['recording_streams']) == 10
    return dict(passed=True,synthetic_participants=22,fit_rows=560,audit_rows=240,
                fit_source_model_count=2,feature_counts_including_intercept=[10,10,19],
                calibration_and_test_covariate_label_poison_changes_no_prefix_inputs=True,
                prefix_prior_blocks_by_fit_participant=8,calibration_chains=4,test_chains=10,
                physical_data_read=False)


if __name__ == '__main__':
    report = dict(scope='Synthetic metadata-only checks; no HARTH archive/cache opened.',
                  adapter=str(PATH),adapter_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),
                  parser_and_lag=parser_check(),prefix=prefix_check())
    output = HERE/'fresh_adapter_review.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),passed=True,scope=report['scope']),indent=2))
