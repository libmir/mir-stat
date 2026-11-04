"""Offline regressions for compiler selection; run with Python's standard library."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import compiler_matrix as matrix


class CompilerMatrixTests(unittest.TestCase):
    def channels(self, ldc="1.43.0", ldc_beta="1.43.0", dmd="2.113.1", dmd_beta="2.114.0-beta.1"):
        return dict(zip(
            (*matrix.CHANNELS["ldc"], *matrix.CHANNELS["dmd"]),
            (ldc, ldc_beta, dmd, dmd_beta),
        ))

    def resolve(self, **versions):
        with redirect_stdout(io.StringIO()):
            return matrix.resolve_compilers(self.channels(**versions).__getitem__)

    def test_duplicate_stable_is_omitted_and_real_beta_is_pinned(self):
        result = self.resolve()
        self.assertEqual(result["compilers"], [
            {"channel": "ldc-latest", "compiler": "ldc-1.43.0"},
            {"channel": "dmd-latest", "compiler": "dmd-2.113.1"},
            {"channel": "dmd-beta", "compiler": "dmd-2.114.0-beta.1"},
        ])
        self.assertEqual(result["stable_compilers"], result["compilers"][:2])

    def test_beta_policy_for_each_family(self):
        cases = [
            ("ldc", "1.43.0-beta1", False),  # Superseded by the final release.
            ("ldc", "1.42.2-rc1", False),
            ("ldc", "1.44.0", False),       # New stable is not a beta.
            ("ldc", "1.44.0-beta1", True),
            ("ldc", "1.43.1-rc2", True),
            ("ldc", "2.0.0-beta1", True),
            ("dmd", "2.113.1-rc.1", False),
            ("dmd", "2.112.9-beta.2", False),
            ("dmd", "2.114.0", False),
            ("dmd", "2.113.2-beta.1", True),
            ("dmd", "2.114.0-rc.1", True),
        ]
        for family, beta, included in cases:
            with self.subTest(family=family, beta=beta):
                result = self.resolve(**{f"{family}_beta": beta})
                selected = [entry["channel"] for entry in result["compilers"]]
                self.assertEqual(f"{family}-beta" in selected, included)

    def test_numeric_order_and_whitespace(self):
        result = self.resolve(ldc="1.9.9\n", ldc_beta=" 1.10.0-beta1\r\n")
        self.assertIn({"channel": "ldc-beta", "compiler": "ldc-1.10.0-beta1"}, result["compilers"])
        self.assertEqual(result["stable_compilers"][0]["compiler"], "ldc-1.9.9")

    def test_no_betas_and_both_betas_keep_stable_coverage(self):
        none = self.resolve(dmd_beta="2.113.0-rc.1")
        self.assertEqual(none["compilers"], none["stable_compilers"])
        self.assertEqual(len(none["compilers"]), 2)
        both = self.resolve(ldc_beta="1.44.0-beta1")
        self.assertEqual([entry["channel"] for entry in both["compilers"]],
                         ["ldc-latest", "ldc-beta", "dmd-latest", "dmd-beta"])
        self.assertEqual(both["stable_compilers"], none["stable_compilers"])

    def test_invalid_channel_data_fails(self):
        for value in ("", "<html>unavailable</html>", "1.43", "1.44.0-dev", "1.44.0-beta1\ncompilers=[]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(ldc_beta=value)
        with self.assertRaises(ValueError):
            self.resolve(dmd="2.114.0-beta.1")
        with self.assertRaises(ValueError):
            self.resolve(dmd_beta="2.114.0-beta1")

    def test_lookup_failure_is_not_a_skip(self):
        def fail(url):
            raise URLError("unavailable")
        with self.assertRaises(URLError):
            matrix.resolve_compilers(fail)

    def test_each_channel_is_resolved_once(self):
        values = self.channels()
        seen = []
        def fetch(url):
            self.assertNotIn(url, seen)
            seen.append(url)
            return values[url]
        with redirect_stdout(io.StringIO()):
            matrix.resolve_compilers(fetch)
        self.assertEqual(set(seen), set(values))

    def test_action_outputs_and_failure_before_publish(self):
        expected = self.resolve()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output"
            with patch.dict(os.environ, GITHUB_OUTPUT=str(path)), patch.object(matrix, "resolve_compilers", return_value=expected):
                matrix.main()
            actual = dict(line.split("=", 1) for line in path.read_text().splitlines())
            self.assertEqual({key: json.loads(value) for key, value in actual.items()}, expected)
            before = path.read_bytes()
            with patch.dict(os.environ, GITHUB_OUTPUT=str(path)), patch.object(matrix, "resolve_compilers", side_effect=URLError("unavailable")):
                with self.assertRaises(URLError):
                    matrix.main()
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
