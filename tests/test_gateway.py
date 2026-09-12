import unittest
import json
import io
from spanda.gateway import SpandaGatewayHandler, GatewayMetrics, GLOBAL_METRICS
from spanda.guardrails import CascadedGuardrail, AuditReceipt
from spanda.integrations.litellm import SpandaLiteLLMGuardrail


class TestGatewayLogic(unittest.TestCase):

    def test_extract_context(self):
        handler = SpandaGatewayHandler.__new__(SpandaGatewayHandler)
        req_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ]
        }
        ctx = handler._extract_context(req_data)
        self.assertEqual(ctx, "You are a helpful assistant.")

    def test_filter_headers(self):
        handler = SpandaGatewayHandler.__new__(SpandaGatewayHandler)
        headers = {
            "Host": "localhost:8080",
            "Authorization": "Bearer sk-12345",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }
        filtered = handler._filter_headers(headers)
        self.assertNotIn("Host", filtered)
        self.assertNotIn("Connection", filtered)
        self.assertEqual(filtered["Authorization"], "Bearer sk-12345")
        self.assertEqual(filtered["Content-Type"], "application/json")

    def test_gateway_metrics(self):
        metrics = GatewayMetrics()
        metrics.record_eval("FAST_PASS_CONSISTENT", 0.0, 1.2, False, is_safe=True)
        metrics.record_eval("HIGH_UNCERTAINTY_REJECT", 0.8, 1.5, True, is_safe=False)
        
        prom_text = metrics.to_prometheus()
        self.assertIn('spanda_evaluations_total{decision="PASSED"} 1', prom_text)
        self.assertIn('spanda_evaluations_total{decision="REJECTED"} 1', prom_text)
        self.assertIn("spanda_mode_collapses_total 1", prom_text)
        self.assertIn("spanda_uptime_seconds", prom_text)

    def test_litellm_guardrail_integration(self):
        guardrail = SpandaLiteLLMGuardrail(threshold=0.35, block_mode=False)
        
        # Mock LiteLLM response object with unanimous agreement
        mock_response = {
            "choices": [
                {"message": {"content": "42"}},
                {"message": {"content": "42.0"}},
                {"message": {"content": "42"}}
            ]
        }
        mock_kwargs = {
            "messages": [
                {"role": "user", "content": "What is the answer?"}
            ]
        }
        
        guardrail.log_success_event(mock_kwargs, mock_response, 0, 1)
        self.assertIn("_spanda_receipt", mock_response)
        receipt = mock_response["_spanda_receipt"]
        self.assertTrue(receipt["is_safe"])
        self.assertIn("PASS", receipt["decision"])
        self.assertAlmostEqual(receipt["rsc"], 0.0, places=2)

    def test_litellm_guardrail_blocking(self):
        guardrail = SpandaLiteLLMGuardrail(threshold=0.35, block_mode=True)
        
        # Mock high disagreement
        mock_response = {
            "choices": [
                {"message": {"content": "Paris"}},
                {"message": {"content": "London"}},
                {"message": {"content": "Berlin"}}
            ]
        }
        mock_kwargs = {"messages": []}
        
        with self.assertRaises(ValueError):
            guardrail.log_success_event(mock_kwargs, mock_response, 0, 1)


if __name__ == "__main__":
    unittest.main()
