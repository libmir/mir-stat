#!/usr/bin/env python3
"""Run the pinned example extractor with package access for internal modules."""

import argparse
from pathlib import Path
import re
import subprocess
import sys


def place_internal_examples(module, examples):
    """Keep public examples external; give internal examples their source package."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
        raise ValueError("invalid source module name")
    parts = module.split(".")
    if "internal" not in parts:
        return examples
    if re.search(r"^\s*module\s+[\w.]+\s*;", examples, re.MULTILINE):
        raise ValueError("extractor unexpectedly supplied a module declaration")
    # Meson's source list includes package.d; its examples belong inside that
    # package, while ordinary modules use the parent package.
    generated = ".".join(parts[:-1] + ["documented_" + parts[-1]])
    return f"module {generated};\n" + examples


def main():
    """Forward extraction unchanged except for the internal module declaration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("extractor")
    parser.add_argument("-i", "--input", required=True, type=Path)
    args = parser.parse_args()
    result = subprocess.run([args.extractor, "-i", str(args.input)],
                            stdout=subprocess.PIPE, text=True, check=True)
    sys.stdout.write(place_internal_examples(args.module, result.stdout))


if __name__ == "__main__":
    main()
