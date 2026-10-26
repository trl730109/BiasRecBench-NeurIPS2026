"""Regression tests for paired robustness scoring; no inference is performed."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RobustnessRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorers = {}
        for domain in ("paper_review", "recruitment"):
            spec = importlib.util.spec_from_file_location(
                f"metrics_{domain}", ROOT / "code" / domain / "evaluate.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.scorers[domain] = module.compute_rr_from_details

    def test_two_failed_predictions_do_not_count_as_agreement(self):
        for domain, score in self.scorers.items():
            for invalid in (None, "", "unparseable"):
                with self.subTest(domain=domain, invalid=invalid):
                    rows = [{"id": "a", "parsed_prediction": invalid}]
                    self.assertEqual(score(rows, rows), "0.00%")

    def test_failure_on_either_side_is_not_agreement(self):
        valid = [{"id": "a", "parsed_prediction": "[[A]]"}]
        failed = [{"id": "a", "parsed_prediction": None}]
        for domain, score in self.scorers.items():
            with self.subTest(domain=domain):
                self.assertEqual(score(valid, failed), "0.00%")
                self.assertEqual(score(failed, valid), "0.00%")

    def test_valid_agreements_and_disagreements_are_unchanged(self):
        for domain, score in self.scorers.items():
            for letter in "ABCDE":
                with self.subTest(domain=domain, letter=letter):
                    rows = [{"id": "a", "parsed_prediction": f"[[{letter}]]"}]
                    self.assertEqual(score(rows, rows), "100.00%")
            with self.subTest(domain=domain, different_answers=True):
                self.assertEqual(score(
                    [{"id": "a", "parsed_prediction": "[[A]]"}],
                    [{"id": "a", "parsed_prediction": "[[B]]"}]), "0.00%")

    def test_failed_pairs_remain_in_the_denominator(self):
        rows = [{"id": "a", "parsed_prediction": "[[C]]"},
                {"id": "b", "parsed_prediction": None}]
        for domain, score in self.scorers.items():
            with self.subTest(domain=domain):
                self.assertEqual(score(rows, rows), "50.00%")

    def test_pairs_are_matched_by_id_not_row_order(self):
        control = [{"id": "a", "parsed_prediction": "[[A]]"},
                   {"id": "b", "parsed_prediction": "[[B]]"},
                   {"id": "control_only", "parsed_prediction": "[[C]]"}]
        injected = [{"id": "b", "parsed_prediction": "[[C]]"},
                    {"id": "a", "parsed_prediction": "[[A]]"},
                    {"id": "injected_only", "parsed_prediction": "[[D]]"}]
        for domain, score in self.scorers.items():
            with self.subTest(domain=domain):
                self.assertEqual(score(control, injected), "50.00%")

    def test_no_paired_samples_is_unavailable(self):
        for domain, score in self.scorers.items():
            with self.subTest(domain=domain):
                self.assertIsNone(score([], []))
                self.assertIsNone(score(
                    [{"id": "a", "parsed_prediction": "[[A]]"}],
                    [{"id": "b", "parsed_prediction": "[[A]]"}]))


if __name__ == "__main__":
    unittest.main()
