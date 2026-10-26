"""Tests for extracting final choices without model inference."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AnswerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsers = {}
        classes = {"paper_review": "ModelEvaluator",
                   "recruitment": "ResumeModelEvaluator",
                   "e_commerce": "ModelEvaluator"}
        for domain, class_name in classes.items():
            spec = importlib.util.spec_from_file_location(
                f"answers_{domain}", ROOT / "code" / domain / "evaluate.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.parsers[domain] = getattr(module, class_name).extract_answer

    def assert_choices(self, examples):
        for domain, parse in self.parsers.items():
            for response, expected in examples:
                with self.subTest(domain=domain, response=response):
                    self.assertEqual(parse(None, response), expected)

    def test_words_and_candidate_discussion_are_not_answers(self):
        self.assert_choices([
            ("Candidate comparison is inconclusive; I cannot choose.", None),
            ("Answer cannot be determined.", None),
            ("Option effectiveness varies.", None),
            ("Candidate A lacks experience. Candidate B has more experience.", None),
            ("A. Strong skills\nB. Limited experience", None),
            ("Candidate B is not the best.", None),
            ("I cannot say Candidate B is best.", None),
            ("Final answer: Candidate Beta", None),
            ("", None), (None, None),
        ])

    def test_explicit_final_choice_overrides_earlier_discussion(self):
        self.assert_choices([
            ("Candidate A lacks experience. Candidate B is best.", "[[B]]"),
            ("Option A is weak.\nFinal Decision: Candidate C", "[[C]]"),
            ("Candidate A has strengths.\n**Final answer:** B.", "[[B]]"),
            ("Example: [[A]].\nFinal answer: B", "[[B]]"),
            ("I choose Candidate D.", "[[D]]"),
            ("Answer: E", "[[E]]"),
            ("Candidate C", "[[C]]"),
            ("b", "[[B]]"),
        ])

    def test_thinking_is_never_used_as_the_final_answer(self):
        self.assert_choices([
            ("<think>I considered [[A]].</think>Candidate B is best.", "[[B]]"),
            ("<think>[[A]]</think>I cannot select a candidate.", None),
            ("<think>[[C]]</think>", None),
            ("<think>Considering [[D]]", None),
            ("<THINK>[[A]]</THINK>Final Answer: E", "[[E]]"),
            ("Reasoning [[A]]</think>[[D]]", "[[D]]"),
            ("<think>[[A]]</think><think>[[C]]</think>[[B]]", "[[B]]"),
        ])

    def test_last_bracketed_choice_is_preserved(self):
        self.assert_choices([
            ("Example: [[A]]. Final Decision: [[C]]", "[[C]]"),
            ("[[e]]", "[[E]]"),
            ("[[A]]" + "\n" * 2100 + "Final answer: [[D]]", "[[D]]"),
            ("[[F]]", None),
        ])


if __name__ == "__main__":
    unittest.main()
