import unittest
from spanda import compute_rsc, detect_hallucination, batch_compute_rsc
from spanda.core import normalize_answer

class TestSpandaCore(unittest.TestCase):

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("  Paris! "), "paris")
        self.assertEqual(normalize_answer("42.0"), "42")  # Canonical float to int
        self.assertEqual(normalize_answer("The Answer is: 100"), "100")  # Prefix stripped
        self.assertEqual(normalize_answer("forty-two"), "42")  # Number words
        self.assertEqual(normalize_answer("$42"), "42")  # Currency stripped
        self.assertEqual(normalize_answer(12345), "12345")
        self.assertEqual(normalize_answer("42.0", canonicalize=False), "420")  # Raw mode

    def test_claude_sonnet_numeric_equivalence(self):
        # Case from Claude Sonnet review
        answers = ["42", "42.0", "The answer is 42", "forty-two", "42"]
        res = compute_rsc(answers)
        self.assertEqual(res["rsc"], 0.0)
        self.assertEqual(res["n_clusters"], 1)
        self.assertEqual(res["dominant_answer"], "42")

    def test_claude_sonnet_polar_agreement(self):
        # Case from Claude Sonnet review
        answers = ["Yes, that is correct.", "Yeah that's right", "Correct", "Indeed, yes", "That is true"]
        res = compute_rsc(answers)
        self.assertEqual(res["rsc"], 0.0)
        self.assertEqual(res["n_clusters"], 1)

    def test_cot_extraction(self):
        answers = ["Therefore #### 42", "\\boxed{42}", "42", "Result: 42"]
        res = compute_rsc(answers)
        self.assertEqual(res["rsc"], 0.0)
        self.assertEqual(res["n_clusters"], 1)

    def test_perfect_consensus(self):
        answers = ["Paris", "paris", "Paris!", " PARIS "]
        res = compute_rsc(answers)
        self.assertEqual(res["rsc"], 0.0)
        self.assertEqual(res["h_norm"], 0.0)
        self.assertEqual(res["w_max"], 1.0)
        self.assertEqual(res["n_clusters"], 1)
        self.assertEqual(res["dominant_answer"], "Paris")

    def test_complete_disagreement(self):
        answers = ["A", "B", "C", "D"]
        res = compute_rsc(answers)
        self.assertAlmostEqual(res["h_norm"], 1.0, places=3)
        self.assertAlmostEqual(res["w_max"], 0.25, places=3)
        self.assertGreater(res["rsc"], 0.8)
        self.assertEqual(res["n_clusters"], 4)

    def test_partial_consensus(self):
        answers = ["Paris", "Paris", "London", "Tokyo"]
        res = compute_rsc(answers)
        self.assertGreater(res["rsc"], 0.0)
        self.assertLess(res["rsc"], 1.0)
        self.assertEqual(res["dominant_answer"], "Paris")
        self.assertEqual(res["n_clusters"], 3)

    def test_detect_hallucination(self):
        confident = ["42", "42", "42", "42"]
        res_conf = detect_hallucination(confident, threshold=0.3)
        self.assertFalse(res_conf["is_uncertain"])

        uncertain = ["10", "20", "30", "40", "50"]
        res_unc = detect_hallucination(uncertain, threshold=0.3)
        self.assertTrue(res_unc["is_uncertain"])

    def test_batch_compute(self):
        batch = [
            ["Paris", "Paris"],
            ["London", "Berlin"]
        ]
        results = batch_compute_rsc(batch)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rsc"], 0.0)
        self.assertGreater(results[1]["rsc"], 0.0)

    def test_empty_input_raises_error(self):
        with self.assertRaises(ValueError):
            compute_rsc([])

if __name__ == "__main__":
    unittest.main()
