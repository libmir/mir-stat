"""Resolve compiler channels once; test only prereleases newer than stable.

Use the channel endpoints consumed by setup-dlang v2.0.0, but pass exact
versions to installation so a channel update cannot change this run's matrix.
Lookup errors and unexpected version formats fail the resolver job.
"""

import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen


CHANNELS = {
    "ldc": (
        "https://ldc-developers.github.io/LATEST",
        "https://ldc-developers.github.io/LATEST_BETA",
    ),
    "dmd": (
        "https://downloads.dlang.org/releases/LATEST",
        "https://downloads.dlang.org/pre-releases/LATEST",
    ),
}


def parse_version(family, text):
    version = text.strip()
    # DMD uses beta.1 / rc.1; LDC uses beta1 / rc1.
    suffix = r"(?:-(?:alpha|beta|rc)\.\d+)?" if family == "dmd" else r"(?:-(?:alpha|beta|rc)\d+)?"
    major = "2" if family == "dmd" else r"\d+"
    match = re.fullmatch(rf"({major})\.(\d+)\.(\d+)({suffix})", version)
    if not match:
        raise ValueError(f"Unexpected {family} version: {version!r}")
    return version, tuple(int(part) for part in match.group(1, 2, 3)), bool(match[4])


def fetch_version(url):
    request = Request(url, headers={"User-Agent": "mir-stat-ci", "Accept": "text/plain"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def resolve_compilers(fetch=fetch_version):
    compilers = []
    stable_compilers = []
    for family, (stable_url, beta_url) in CHANNELS.items():
        stable, stable_number, stable_prerelease = parse_version(family, fetch(stable_url))
        beta, beta_number, beta_prerelease = parse_version(family, fetch(beta_url))
        if stable_prerelease:
            raise ValueError(f"{family} stable channel returned a prerelease: {stable}")
        entry = {"channel": f"{family}-latest", "compiler": f"{family}-{stable}"}
        stable_compilers.append(entry)
        compilers.append(entry)
        print(f"{family}-latest: {stable}")
        # A prerelease of the same major.minor.patch is older than its stable
        # release; no prerelease-suffix ordering is needed for this comparison.
        if beta_prerelease and beta_number > stable_number:
            compilers.append({"channel": f"{family}-beta", "compiler": f"{family}-{beta}"})
            print(f"{family}-beta: {beta} (newer prerelease; included)")
        else:
            print(f"{family}-beta: {beta} (no newer prerelease; omitted)")
    return {"compilers": compilers, "stable_compilers": stable_compilers}


def main():
    matrices = resolve_compilers()
    # Resolve and validate every channel before publishing any matrix output.
    output = "".join(f"{name}={json.dumps(value, separators=(',', ':'))}\n"
                     for name, value in matrices.items())
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
