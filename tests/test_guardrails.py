import unittest
from spanda import CascadedGuardrail, AuditReceipt

class TestCascadedGuardrail(unittest.TestCase):

    def setUp(self):
        self.guard = CascadedGuardrail(
            uncertainty_threshold=0.3,
            grounding_threshold=0.2
        )

    def test_tier1_early_exit_on_high_entropy(self):
        # 4 conflicting answers -> high entropy -> Tier 1 rejection
        conflicting_samples = ["10", "25", "40", "99"]
        receipt = self.guard.evaluate(conflicting_samples)
        
        self.assertFalse(receipt.is_safe)
        self.assertEqual(receipt.decision, "UNCERTAIN_HIGH_ENTROPY")
        self.assertEqual(receipt.tier_executed, 1)
        self.assertGreater(receipt.rsc, 0.3)
        self.assertLess(receipt.latency_ms, 10.0)  # sub-10ms

    def test_tier1_fast_pass_without_context(self):
        # Unanimous samples, no context check requested -> Fast Pass
        consistent = ["paris", "Paris", "PARIS"]
        receipt = self.guard.evaluate(consistent)

        self.assertTrue(receipt.is_safe)
        self.assertEqual(receipt.decision, "FAST_PASS_CONSISTENT")
        self.assertEqual(receipt.tier_executed, 1)
        self.assertEqual(receipt.rsc, 0.0)

    def test_tier2_grounded_safe(self):
        # Strong consensus AND grounded in context -> Verified Safe
        answers = ["Acme Corp 2003", "Acme Corp 2003", "Acme Corp 2003", "Acme Corp 2003"]
        context = "Company background: Acme Corp was officially founded in 2003 in California."
        
        receipt = self.guard.evaluate(answers, context=context)
        self.assertTrue(receipt.is_safe)
        self.assertEqual(receipt.decision, "VERIFIED_GROUNDED_SAFE")
        self.assertEqual(receipt.tier_executed, 2)
        self.assertIsNotNone(receipt.grounding_overlap)
        self.assertGreater(receipt.grounding_overlap, 0.2)

    def test_tier2_catches_confident_mode_collapse(self):
        # Unanimous consensus (R_sc = 0.0) BUT 0% grounding in source context -> Mode Collapse Alert!
        hallucinated_unanimous = [
            "George Washington",
            "George Washington",
            "George Washington",
            "George Washington"
        ]
        context = "The document discusses quantum entanglement and superconducting qubits."

        receipt = self.guard.evaluate(hallucinated_unanimous, context=context)
        self.assertFalse(receipt.is_safe)
        self.assertEqual(receipt.decision, "MODE_COLLAPSE_RISK")
        self.assertEqual(receipt.tier_executed, 2)
        self.assertEqual(receipt.rsc, 0.0)  # Unanimous, but false!
        self.assertEqual(receipt.grounding_overlap, 0.0)

    def test_tool_call_argument_consensus(self):
        # Matching tool calls -> Safe
        valid_tool_calls = [
            {"command": "pytest tests/", "timeout": 30},
            {"command": "pytest tests/", "timeout": 30},
            {"command": "pytest tests/", "timeout": 30},
        ]
        receipt = self.guard.evaluate_tool_calls(valid_tool_calls)
        self.assertTrue(receipt.is_safe)
        self.assertEqual(receipt.decision, "TOOL_ARG_VERIFIED")

        # Conflicting tool calls (e.g. different dangerous commands) -> Blocked
        dangerous_tool_calls = [
            {"command": "rm -rf /tmp/cache", "target": "cache"},
            {"command": "rm -rf /var/log", "target": "logs"},
            {"command": "ls -la", "target": "root"},
        ]
        receipt_danger = self.guard.evaluate_tool_calls(dangerous_tool_calls)
        self.assertFalse(receipt_danger.is_safe)
        self.assertEqual(receipt_danger.decision, "TOOL_ARG_MISMATCH")

    def test_audit_receipt_serialization(self):
        receipt = self.guard.evaluate(["42", "42"])
        receipt_dict = receipt.to_dict()
        self.assertIn("decision", receipt_dict)
        self.assertIn("latency_ms", receipt_dict)
        self.assertIn("confidence", receipt_dict)

if __name__ == "__main__":
    unittest.main()
