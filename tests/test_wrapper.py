"""
Unit tests for Spanda 1-Line Universal Client Wrapper.
"""

import unittest
from unittest.mock import MagicMock
import spanda
from spanda.wrapper import wrap, SpandaUncertaintyError


class MockChoice:
    def __init__(self, content: str):
        self.message = MagicMock(content=content)
        self.text = content


class MockResponse:
    def __init__(self, contents):
        self.choices = [MockChoice(c) for c in contents]


class MockChatCompletions:
    def __init__(self, contents):
        self._contents = contents

    def create(self, *args, **kwargs):
        return MockResponse(self._contents)


class MockClient:
    def __init__(self, contents):
        self.chat = MagicMock()
        self.chat.completions = MockChatCompletions(contents)


class TestSpandaWrapper(unittest.TestCase):

    def test_unanimous_consensus_wrapper(self):
        # 3 identical paths
        client = MockClient(["42", "42.0", "42"])
        wrapped = wrap(client, k=3, threshold=0.35, block=False)

        resp = wrapped.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            n=1
        )

        self.assertTrue(hasattr(resp, "spanda"))
        self.assertEqual(resp.spanda.rsc, 0.0)
        self.assertTrue(resp.spanda.is_safe)
        self.assertIn("PASS", resp.spanda.decision)
        # Dominant choice collapsed back to 1
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].text, "42")

    def test_disagreement_detection(self):
        # 3 contradictory answers
        client = MockClient(["Paris", "London", "Berlin"])
        wrapped = wrap(client, k=3, threshold=0.35, block=False)

        resp = wrapped.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Where is it?"}],
            n=1
        )

        self.assertTrue(hasattr(resp, "spanda"))
        self.assertFalse(resp.spanda.is_safe)
        self.assertGreater(resp.spanda.rsc, 0.5)

    def test_block_mode_raises_error(self):
        client = MockClient(["Apple", "Banana", "Orange"])
        wrapped = wrap(client, k=3, threshold=0.35, block=True)

        with self.assertRaises(SpandaUncertaintyError) as ctx:
            wrapped.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Pick a fruit"}],
                n=1
            )
        self.assertIn("Spanda guardrail blocked completion", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
