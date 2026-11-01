#!/usr/bin/env python3
"""Wire the Mir Meson template through the package-aware example extractor."""

import argparse
from pathlib import Path


def configure_extraction(template):
    """Replace exactly one extractor invocation, rejecting unfamiliar templates."""
    command = "                'dtools_tests_extractor',"
    if template.count(command) != 1:
        raise ValueError("expected one extractor command in the Mir Meson template")
    return template.replace(command,
        "                'python3', join_paths(root, 'tools', 'extract_doc_examples.py'),\n"
        "                '--module', source.replace('/', '.'),\n" + command)


def main():
    """Adapt the copied template before CircleCI appends it to meson.build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    args = parser.parse_args()
    template = args.template.read_text(encoding="utf-8")
    args.template.write_text(configure_extraction(template), encoding="utf-8")


if __name__ == "__main__":
    main()
