# Reconstructing caches from the original acquisitions

The unchanged adapter contains the full feature conversion: MHEALTH selects every
50th acquisition row, retains activity labels1–12 and the prescribed21 non-ECG
channels; HAR computes mean/SD/min/max of the prescribed six128-reading channels,
retains labels and participant IDs, and constructs the fixed participant/segment
order. The cache stores the resulting exact float64 input matrix before any
prefix-dependent scaling or model fitting.

Verify that official raw files produce the supplied caches:

```sh
python prepare_raw_data.py --data full_raw_data
python verify_cache_from_raw.py --raw-data full_raw_data
```

Produce a separate rebuilt input folder and verify it, preserving supplied data:

```sh
python verify_cache_from_raw.py --raw-data full_raw_data --output rebuilt_data --report rebuilt_cache_verification.json
```

`verify_cache_from_raw.py` verifies original consumed-file hashes, every array value/dtype,
labels/order/participants and all initially fitted models, priors, qualities and
prefix transformations. It compares values exactly, rather than rounding floating
point. It also reports NPZ byte SHA equality. Across compression-library versions,
compressed bytes may differ even if all scientific input arrays are identical;
the rebuilt metadata then records its own new byte hash. No frozen algorithm file
is modified.

The rebuilt wearable inputs can then be used with the same protocol and runner:

```sh
mkdir -p rerun_rebuilt_wearable
cp new_bayes_extension/protocol.json rerun_rebuilt_wearable/protocol.json
python cached_runner.py independent_bayes_extension.py --data rebuilt_data --output rerun_rebuilt_wearable
```

Full occupancy inputs remain in the supplied original CSV/TXT directories or
`full_raw_data`; their raw adapter does not require NPZ conversion.
