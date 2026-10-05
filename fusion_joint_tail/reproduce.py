"""Fresh-output launcher; retains the project's frozen scientific bindings."""
from pathlib import Path
import argparse
import shutil
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
PROJECT = SOURCE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "verify", "daphnet-calibrate",
                                         "daphnet-test", "rss-calibrate", "rss-test"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == SOURCE or SOURCE in output.parents:
        parser.error("Choose a fresh directory outside the delivered research folder.")
    if args.phase == "init":
        if output.exists():
            parser.error("The output directory must not exist for init.")
        output.mkdir(parents=True)
        for name in ("tail_fusion.py", "run_replay.py", "physical_adapter.py",
                     "verify_tail.py", "protocol.json", "pre_acquisition_freeze.json",
                     "protocol_text_clarification.json"):
            shutil.copy2(SOURCE / name, output / name)
        shutil.copytree(SOURCE / "data", output / "data")
        print(f"Initialized {output}; project bindings remain at {PROJECT}")
        return
    if not (output / "pre_acquisition_freeze.json").is_file():
        parser.error("Run init on this fresh output directory first.")
    # Load the original frozen sources. Only output/cache locations change.
    import run_replay as api
    api.verify()
    api.ROOT = output
    api.verify()
    if args.phase == "verify":
        metadata = api.load(output / "data/cache_metadata.json")
        assert api.sha(output / "data/cached_dataset.npz") == metadata["cache_sha256"]
        print("Original frozen sources, protocol and fresh data cache verified.")
    elif args.phase == "daphnet-calibrate":
        api.calibrate()
    elif args.phase == "daphnet-test":
        api.test()
    else:
        import run_rss_development as rss
        rss.ROOT = output / "rss_development"
        rss.ROOT.mkdir(exist_ok=True)
        {"rss-calibrate": rss.calibrate, "rss-test": rss.test}[args.phase]()


if __name__ == "__main__":
    main()
