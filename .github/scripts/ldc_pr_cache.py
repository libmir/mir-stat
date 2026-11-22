"""Combine master and PR LDC objects for a build and collect the PR delta."""

import argparse
import json
import os
from pathlib import Path
import shutil


def objects(directory):
    suffix = ".obj" if os.name == "nt" else ".o"
    return {path.name for path in directory.glob(f"ircache_*{suffix}") if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect"))
    args = parser.parse_args()

    root = Path(os.environ["RUNNER_TEMP"])
    build = root / os.environ["CACHE_DIR"]
    pr = root / os.environ["PR_CACHE_DIR"]
    manifest = root / (build.name + "-baseline.json")

    if args.phase == "prepare":
        build.mkdir(exist_ok=True)
        pr.mkdir(exist_ok=True)
        baseline = objects(build)
        manifest.write_text(json.dumps(sorted(baseline)), encoding="utf-8")
        for name in objects(pr) - baseline:
            shutil.copyfile(pr / name, build / name)
        return

    baseline = set(json.loads(manifest.read_text(encoding="utf-8")))
    previous = objects(pr)
    current = objects(build) - baseline
    for name in previous - current:
        (pr / name).unlink()
    for name in current - previous:
        (build / name).replace(pr / name)
    print(f"LDC cache: {len(baseline)} master objects, {len(previous)} previous PR "
          f"objects, {len(current)} current PR objects.")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"changed={str(previous != current).lower()}", file=output)


if __name__ == "__main__":
    main()
