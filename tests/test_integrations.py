import unittest
from spanda.integrations.langchain import SpandaStringEvaluator, SpandaGuardrailRunnable
from spanda.integrations.llamaindex import SpandaRAGGuardrail

class TestIntegrations(unittest.TestCase):

    def test_langchain_evaluator_pass(self):
        evaluator = SpandaStringEvaluator(uncertainty_threshold=0.35)
        samples = ["Paris", "Paris", "Paris", "paris.", "Paris"]
        res = evaluator.evaluate_strings(
            prediction=samples,
            context="Paris is the capital of France."
        )
        self.assertEqual(res["value"], "PASS")
        self.assertEqual(res["dominant_answer"], "Paris")
        self.assertLess(res["score"], 0.2)
        self.assertIn("audit_receipt", res)

    def test_langchain_evaluator_hallucination(self):
        evaluator = SpandaStringEvaluator(uncertainty_threshold=0.35)
        samples = ["Rome", "Berlin", "Madrid", "Lisbon", "Paris"]
        res = evaluator.evaluate_strings(prediction=samples)
        self.assertEqual(res["value"], "HALLUCINATION_RISK")
        self.assertGreater(res["score"], 0.5)

    def test_langchain_runnable_invoke(self):
        runnable = SpandaGuardrailRunnable(uncertainty_threshold=0.35)
        res = runnable.invoke(["Berlin", "Berlin", "Berlin", "Berlin"])
        self.assertTrue(res["is_safe"])
        self.assertEqual(res["output"], "Berlin")

    def test_llamaindex_guardrail_validation(self):
        rag_guard = SpandaRAGGuardrail(uncertainty_threshold=0.3)
        receipt = rag_guard.validate_response(
            samples=["Albert Einstein", "Albert Einstein", "Albert Einstein"],
            context_str="Albert Einstein was a theoretical physicist."
        )
        self.assertTrue(receipt.is_safe)
        self.assertEqual(receipt.dominant_answer, "Albert Einstein")

if __name__ == "__main__":
    unittest.main()
