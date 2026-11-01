"""Regression checks for build snapshots and Meson test-result validation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from build_docs import copy_build_inputs, validate_test_results
from extract_doc_examples import place_internal_examples
from prepare_doc_examples import configure_extraction


class TestResultsTest(unittest.TestCase):
    def test_example_mode_requires_every_configured_test(self):
        names = ["mir-stat-test",
                 "mir/stat/distribution/normal documented unittest", "mir/stat/transform documented unittest"]
        configured = [{"name": name} for name in names]
        results = [{"name": name, "result": "OK"} for name in names]
        validate_test_results(results, configured)
        with self.assertRaisesRegex(RuntimeError, "match"):
            validate_test_results(results[:-1], configured)
        with self.assertRaisesRegex(RuntimeError, "match"):
            validate_test_results(results + [{"name": "unexpected", "result": "OK"}], configured)

    def test_example_mode_rejects_missing_examples(self):
        names = ["mir-stat-test"]
        with self.assertRaisesRegex(RuntimeError, "no extracted"):
            validate_test_results([{"name": name, "result": "OK"} for name in names],
                                  [{"name": name} for name in names])

    def test_accepts_additional_passing_tests(self):
        validate_test_results([
            {"name": "mir-stat-test", "result": "OK"},
            {"name": "additional-test", "result": "OK"},
        ])

    def test_requires_library_test(self):
        for names in ([], ["additional-test"]):
            with self.subTest(names=names), self.assertRaisesRegex(RuntimeError, "library test"):
                validate_test_results([{"name": name, "result": "OK"} for name in names])

    def test_rejects_unsuccessful_results_anywhere_in_suite(self):
        names = ["mir-stat-test", "additional-test"]
        for unsuccessful in names:
            for result in ("FAIL", "TIMEOUT", "SKIP"):
                with self.subTest(name=unsuccessful, result=result):
                    with self.assertRaisesRegex(RuntimeError, "all Meson tests"):
                        validate_test_results([
                            {"name": name, "result": result if name == unsuccessful else "OK"}
                            for name in names
                        ])


class BuildInputsTest(unittest.TestCase):
    def test_working_tree_changes_and_staged_deletions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()

            def git(*args):
                return subprocess.run(["git", *args], cwd=repo, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            git("init")
            (repo / "source").mkdir()
            for name in ("edited.d", "deleted.d", "untracked-later.d", "missing.d"):
                (repo / "source" / name).write_text("original\n")
            (repo / "meson.build").write_text("original\n")
            git("add", ".")
            # This commit exists only inside the temporary test fixture.
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-m", "Fixture")
            snapshot = root / "snapshot"
            git("clone", "--no-hardlinks", str(repo), str(snapshot))

            (repo / "source/edited.d").write_text("edited\n")
            (repo / "meson.build").write_text("edited build\n")
            git("rm", "source/deleted.d")
            git("rm", "--cached", "source/untracked-later.d")
            (repo / "source/missing.d").unlink()
            (repo / "source/added.d").write_text("added\n")
            git("add", "source/added.d")
            (repo / "source/scratch.d").write_text("untracked\n")

            copy_build_inputs(repo, snapshot)

            self.assertEqual((snapshot / "source/edited.d").read_text(), "edited\n")
            self.assertEqual((snapshot / "meson.build").read_text(), "edited build\n")
            self.assertEqual((snapshot / "source/added.d").read_text(), "added\n")
            for name in ("deleted.d", "untracked-later.d", "missing.d", "scratch.d"):
                self.assertFalse((snapshot / "source" / name).exists(), name)
            self.assertTrue((repo / "source/untracked-later.d").is_file())
            self.assertTrue((repo / "source/scratch.d").is_file())


class ExampleTemplateTest(unittest.TestCase):
    def test_wraps_extractor_without_changing_arguments(self):
        command = "                'dtools_tests_extractor',"
        original = "command: [\n" + command + "\n                '-i', '@INPUT@'],\n"
        adapted = configure_extraction(original)
        self.assertIn("'python3', join_paths(root, 'tools', 'extract_doc_examples.py')", adapted)
        self.assertIn("'--module', source.replace('/', '.')", adapted)
        self.assertTrue(adapted.endswith(command + "\n                '-i', '@INPUT@'],\n"))

    def test_rejects_missing_or_multiple_extractor_commands(self):
        for template in ("", "                'dtools_tests_extractor',\n" * 2):
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "expected one"):
                configure_extraction(template)


class ExamplePackageTest(unittest.TestCase):
    def test_public_examples_remain_external(self):
        text = "# line 10\nunittest { import mir.stat.distance; }\n"
        self.assertEqual(place_internal_examples("mir.stat.distance", text), text)
        self.assertEqual(place_internal_examples("mir.stat.package", text), text)

    def test_internal_package_and_leaf_examples(self):
        text = "# line 161\nunittest { import mir.stat.internal.linearAlgebra; }\n"
        for source, generated in [
            ("mir.stat.internal.linearAlgebra.package", "mir.stat.internal.linearAlgebra.documented_package"),
            ("mir.stat.internal.linearAlgebra.products", "mir.stat.internal.linearAlgebra.documented_products"),
            ("mir.math.internal.benchmark", "mir.math.internal.documented_benchmark"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(place_internal_examples(source, text), f"module {generated};\n" + text)

    def test_internal_is_a_package_component(self):
        self.assertEqual(place_internal_examples("mir.stat.internalized", "unittest {}"), "unittest {}")

    def test_reject_unexpected_extractor_module(self):
        with self.assertRaisesRegex(ValueError, "unexpectedly"):
            place_internal_examples("mir.stat.internal.linearAlgebra.package", "module external;\n")

    def test_reject_invalid_module_name(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            place_internal_examples("mir/stat/internal/products", "")


if __name__ == "__main__":
    unittest.main()
