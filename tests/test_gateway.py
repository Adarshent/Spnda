import unittest
import json
import io
from spanda.gateway import SpandaGatewayHandler
from spanda.guardrails import CascadedGuardrail

class DummyServer:
    def __init__(self):
        pass

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

if __name__ == "__main__":
    unittest.main()
