"""Compare the two LDC coverage artifacts from the Ubuntu cache experiment."""

import sys
from pathlib import Path


def compare(left, right):
    names = {p.name for p in left.glob("source-*.lst")}
    other_names = {p.name for p in right.glob("source-*.lst")}
    if not names or names != other_names:
        raise ValueError(f"report inventory mismatch or empty inventory: "
                         f"master-only={sorted(names - other_names)}, "
                         f"master-plus-pr={sorted(other_names - names)}")

    changed_counts = 0
    for name in sorted(names):
        before = (left / name).read_bytes().splitlines()
        after = (right / name).read_bytes().splitlines()
        if len(before) != len(after):
            raise ValueError(f"{name}: report length changed")
        for line, (old, new) in enumerate(zip(before, after), 1):
            if old == new:
                continue
            old_count, sep, old_text = old.partition(b"|")
            new_count, _, new_text = new.partition(b"|")
            a, b = old_count.strip(), new_count.strip()
            if not (sep and old_text == new_text and a.isdigit() and b.isdigit()
                    and int(a) > 0 and int(b) > 0):
                raise ValueError(f"{name}:{line}: coverage or source changed\n"
                                 f"  master-only:    {old!r}\n  master-plus-pr: {new!r}")
            print(f"{name}:{line}: {int(a)} -> {int(b)} hits")
            changed_counts += 1
    print(f"{len(names)} reports have matching line coverage; "
          f"{changed_counts} positive execution counts differ.")


if __name__ == "__main__":
    try:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    except ValueError as error:
        sys.exit(str(error))
