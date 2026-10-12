"""Behavioral checks for leakage, annotation policy, thresholds and inference."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_quality import message_key, parse_labels, prepare_data
from evaluation import choose_thresholds, probabilities
from experiment import candidates, split_frames
from inference import predict_message


class DataPolicyTests(unittest.TestCase):
    def test_cosmetic_duplicates_preserve_negation(self):
        self.assertEqual(message_key("  WATER\nneeded "), message_key("water needed"))
        self.assertNotEqual(message_key("need water"), message_key("do not need water"))

    def test_conflicts_ambiguous_labels_and_empty_are_quarantined(self):
        messages = pd.DataFrame({"id": range(1, 9), "message": ["help", "HELP", "food", "water", "water", " ", "rescue", "normal"], "genre": "direct"})
        categories = pd.DataFrame({"id": [1, 2, 3, 3, 4, 5, 6, 7, 8],
                                   "categories": ["related-1;food-0", "related-1;food-0", "related-0;food-0", "related-1;food-1",
                                                  "related-1;food-0", "related-1;food-1", "related-1;food-0", "related-2;food-0", "related-0;food-0"]})
        clean, names, audit, rejected = prepare_data(messages, categories)
        self.assertEqual(clean.id.tolist(), [1, 8])
        self.assertEqual(names, ["related", "food"])
        self.assertEqual(audit["conflicting_ids"], 1)
        self.assertEqual(audit["conflicting_text_groups"], 1)
        self.assertEqual(audit["consistent_text_duplicates"], 1)
        self.assertIn("related_2_meaning_unverified", set(rejected.reason))
        self.assertEqual(len(messages), 8)  # callers' raw inputs were not mutated

    def test_invalid_label_schema_fails(self):
        with self.assertRaises(ValueError):
            parse_labels(pd.Series(["related-1;food-0", "food-1;related-0"]))
        with self.assertRaises(ValueError):
            parse_labels(pd.Series(["related-1;food-9"]))

    def test_split_is_deterministic_and_keeps_repeated_groups_together(self):
        frame = pd.DataFrame({"id": range(120), "message_group": [f"group-{i // 2}" for i in range(120)]})
        first, second = split_frames(frame), split_frames(frame)
        for name in first:
            self.assertEqual(first[name].id.tolist(), second[name].id.tolist())
        self.assertEqual(sum(len(v) for v in first.values()), 120)
        groups = [set(v.message_group) for v in first.values()]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])


class ModelPolicyTests(unittest.TestCase):
    def test_threshold_selection_support_and_ties(self):
        truth = np.array([[1, 1], [1, 0], [0, 0], [0, 0]])
        scores = np.array([[.35, .7], [.4, .1], [.1, .2], [.15, .3]])
        table = choose_thresholds(truth, scores, ["supported", "rare"], min_positives=2)
        self.assertLess(table.iloc[0].threshold, .5)
        self.assertEqual(table.iloc[1].threshold, .5)
        self.assertFalse(table.iloc[1].tuned)

    def test_unseen_words_never_enter_training_vocabulary(self):
        pipeline = candidates(False)["logistic_c1"]
        x = np.array(["water needed", "water help", "food needed", "food help", "sunny today", "sunny outside"])
        y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
        pipeline.fit(x, y)
        pipeline.predict(["validationonlyword"])
        self.assertNotIn("validationonlyword", pipeline.named_steps["tfidf"].vocabulary_)
        scores = probabilities(pipeline, x)
        self.assertEqual(scores.shape, (6, 2))
        bundle = {"model": pipeline, "category_names": ["water", "food"], "thresholds": [.0, 1.0],
                  "metadata": {"run_id": "test", "selected_model": "logistic", "excluded_labels": []}}
        result = predict_message(bundle, "water help")
        self.assertEqual(result["categories"], ["water"])
        self.assertEqual(predict_message(bundle, "zzzzqqqq")['scores'], [])
        with self.assertRaises(ValueError):
            predict_message(bundle, "  ")
        with self.assertRaises(ValueError):
            predict_message(bundle, "x" * 5001)


if __name__ == "__main__":
    unittest.main()
