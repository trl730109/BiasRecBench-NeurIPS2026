"""Offline benchmark loading and CLI checks; no inference calls are made."""

import ast
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from datasets import load_dataset
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("paper_review", "recruitment", "e_commerce")
COLUMNS = {"id", "bias_type", "input", "target_output_bias", "target_output_clean", "meta"}


class ReleaseTests(unittest.TestCase):
    def cli(self, script, *args):
        self.assertTrue(script.is_file(), f"Missing CLI: {script}")
        env = os.environ.copy()
        for key in ("API_KEY", "API_KEYS", "API_BASE", "SYNTH_MODEL_NAME", "api_keys", "api_base"):
            env.pop(key, None)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(script), *args], cwd=cwd,
                                    env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_api_entry_points_do_not_require_the_optional_gpu_backend(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                self.assertIn("--input_dir", self.cli(
                    ROOT / "code" / domain / "evaluate.py", "--help"))

    def test_generation_stages_offer_help_without_api_credentials(self):
        for domain in DOMAINS:
            directory = ROOT / "code" / domain
            with self.subTest(domain=domain, stage="build"):
                self.assertIn("--num_samples", self.cli(directory / "build_dataset.py", "--help"))
            if domain != "recruitment":
                with self.subTest(domain=domain, stage="prepare"):
                    self.assertIn("--output_dir", self.cli(
                        directory / "build_dataset.py", "--stage", "prepare", "--help"))
            for stage in ("inject", "refine", "control"):
                with self.subTest(domain=domain, stage=stage):
                    self.assertIn("usage:", self.cli(
                        directory / "inject_biases.py", "--stage", stage, "--help").lower())

    def test_llm_criterion_validation_cli_is_available(self):
        self.assertIn("--criteria", self.cli(ROOT / "code" / "validate_criteria.py", "--help"))

    def load_splits(self, domain, input_dir):
        path = ROOT / "code" / domain / "evaluate.py"
        self.assertTrue(path.is_file(), str(path))
        tree = ast.parse(path.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and "__name__" in ast.unparse(node.test))
        start = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "parquet_files" for t in node.targets))
        stop = next(i for i in range(start, len(main.body))
                    if isinstance(main.body[i], ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "dataset_dict" for t in main.body[i].targets))
        block = ast.Module(body=main.body[start:stop + 1], type_ignores=[])
        scope = {"args": SimpleNamespace(input_dir=str(input_dir)),
                 "os": os, "re": re, "glob": glob, "load_dataset": load_dataset}
        exec(compile(block, str(path), "exec"), scope)
        return scope["dataset_dict"]

    def test_each_evaluator_loads_its_bundled_parquet_splits(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                splits = self.load_splits(domain, ROOT / "data" / domain)
                self.assertEqual(len(splits), 15)
                for data in splits.values():
                    self.assertGreater(len(data), 0)
                    self.assertEqual(set(data.column_names), COLUMNS)
                    self.assertRegex(data[0]["target_output_clean"], r"^\[\[[A-E]\]\]$")
                for condition in ("biased_first", "biased_last"):
                    mapping = ROOT / "data" / domain / "position" / f"mapping_Position_{condition}.jsonl"
                    self.assertTrue(mapping.is_file(), str(mapping))
                    self.assertGreater(len(mapping.read_text().splitlines()), 0)

    def test_run_script_defaults_to_eval_with_original_settings_from_any_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python))
            for domain in DOMAINS:
                with self.subTest(domain=domain):
                    script = ROOT / "code" / domain / "run.sh"
                    self.assertTrue(script.is_file(), str(script))
                    result = subprocess.run(["bash", str(script)], cwd=directory, env=env,
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    argv = json.loads(result.stdout.splitlines()[-1])
                    self.assertEqual(argv[0], str(script.parent / "evaluate.py"))
                    for key, value in (("--input_dir", str(ROOT / "data" / domain)),
                                       ("--mapping_dir", str(ROOT / "data" / domain / "position")),
                                       ("--temperature", "0.7"), ("--max_tokens", "4096"),
                                       ("--tensor_parallel_size", "4"), ("--batch_size", "50")):
                        self.assertEqual(argv[argv.index(key) + 1], value)

    def test_evaluators_discover_paired_conditions_in_bias_subfolders(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as directory:
                bias = "Bandwagon" if domain == "e_commerce" else "Authority"
                nested = Path(directory) / bias.lower()
                nested.mkdir()
                expected_ids = {}
                for condition in ("biased", "unbiased"):
                    name = f"{bias}_eval_{bias}_{condition}-00000-of-00001.parquet"
                    source = next((ROOT / "data" / domain).rglob(name))
                    expected_ids[condition] = pq.read_table(source, columns=["id"])["id"].to_pylist()
                    shutil.copyfile(source, nested / name)
                try:
                    splits = self.load_splits(domain, directory)
                except FileNotFoundError:
                    self.fail("Evaluator did not discover the bias subfolder's parquet conditions")
                self.assertEqual(set(splits), {f"{bias}_eval_{bias}_biased", f"{bias}_eval_{bias}_unbiased"})
                for condition in ("biased", "unbiased"):
                    self.assertEqual(list(splits[f"{bias}_eval_{bias}_{condition}"]["id"]),
                                     expected_ids[condition])

    def test_run_script_routes_generation_actions_and_margin_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python))
            pools = {"paper_review": "selector_dataset_raw", "recruitment": "selector_resume_raw",
                     "e_commerce": "selector_shopping_raw"}
            for domain in DOMAINS:
                script = ROOT / "code" / domain / "run.sh"
                for action in ("build", "inject", "refine", "control"):
                    with self.subTest(domain=domain, action=action):
                        extra = ["--epsilon", "2"] if action in ("build", "inject") and domain != "paper_review" else []
                        result = subprocess.run(["bash", str(script), action, *extra], cwd=directory,
                                                env=env, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        argv = json.loads(result.stdout.splitlines()[-1])
                        expected = "build_dataset.py" if action == "build" else "inject_biases.py"
                        self.assertEqual(argv[0], str(script.parent / expected))
                        if action in ("refine", "control"):
                            self.assertEqual(argv[1:3], ["--stage", action])
                        if extra:
                            flag = "--output_file" if action == "build" else "--input_file"
                            expected_pool = ROOT / "outputs" / domain / "datasets" / f"{pools[domain]}_eps2.jsonl"
                            self.assertEqual(argv[argv.index(flag) + 1], str(expected_pool))
                            if action == "inject":
                                self.assertEqual(argv[argv.index("--output_dir") + 1],
                                                 str(ROOT / "outputs" / domain / "biased_datasets_eps2"))

    def test_paper_pool_build_and_static_injection_write_aligned_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gold = directory / "gold.jsonl"
            rejects = directory / "rejects.jsonl"
            pools = directory / "pools.jsonl"
            gold.write_text(json.dumps({"title": "Gold", "abstract": "Gold abstract", "rates": [9, 9]}) + "\n")
            rejects.write_text("".join(json.dumps({"title": f"Reject {i}", "abstract": f"Abstract {i}",
                                                   "rates": [7, 7]}) + "\n" for i in range(4)))
            self.cli(ROOT / "code/paper_review/build_dataset.py", "--spotlight_path", str(gold),
                     "--reject_path", str(rejects), "--output_file", str(pools), "--num_samples", "1")
            pool = json.loads(pools.read_text())
            self.assertEqual(len(pool["papers"]), 5)
            self.assertEqual(pool["papers"][pool["gold_index"]]["role"], "gold")
            output = directory / "injected"
            result = subprocess.run([sys.executable, str(ROOT / "code/paper_review/inject_biases.py"),
                                     "--input_file", str(pools), "--output_dir", str(output),
                                     "--bias_types", "Instruction"], cwd=directory, capture_output=True,
                                    text=True, timeout=30, env=dict(os.environ, API_KEY="offline-test",
                                    API_BASE="http://127.0.0.1:1", SYNTH_MODEL_NAME="offline-test"))
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads((output / "eval_Instruction.jsonl").read_text())
            self.assertEqual(row["id"], "sample_0")
            self.assertEqual(row["bias_type"], "Instruction")
            self.assertNotEqual(row["target_output_clean"], row["target_output_bias"])
            self.assertEqual(set(re.findall(r"\[Candidate ([A-E])\]", row["input"])), set("ABCDE"))


if __name__ == "__main__":
    unittest.main()
