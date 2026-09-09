"""
LiteLLM Integration for Spanda (BRHMN Labs).

Enables drop-in epistemic uncertainty quantification and 2-tier cascaded
guardrails within LiteLLM proxy and SDK pipelines.

Usage:
    import litellm
    from spanda.integrations.litellm import SpandaLiteLLMGuardrail

    # Register Spanda as a callback
    litellm.callbacks = [SpandaLiteLLMGuardrail(threshold=0.35, block_mode=False)]

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 17 * 19?"}],
        n=3
    )
    print(response._spanda_receipt)
"""

from typing import Optional, List, Dict, Any
from spanda.guardrails import CascadedGuardrail, AuditReceipt

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:
    # Graceful fallback if litellm is not installed in local environment
    class CustomLogger:  # type: ignore
        pass


class SpandaLiteLLMGuardrail(CustomLogger):
    """
    LiteLLM custom callback for sub-millisecond hallucination and mode-collapse detection.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        grounding_threshold: float = 0.15,
        block_mode: bool = False,
        default_k: int = 3,
    ):
        self.guardrail = CascadedGuardrail(
            uncertainty_threshold=threshold,
            grounding_threshold=grounding_threshold,
        )
        self.block_mode = block_mode
        self.default_k = default_k

    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any):
        """Hook called by LiteLLM after completion succeeds."""
        choices = getattr(response_obj, "choices", None)
        if not choices:
            if isinstance(response_obj, dict):
                choices = response_obj.get("choices", [])
            else:
                return

        samples: List[str] = []
        for c in choices:
            if isinstance(c, dict):
                text = c.get("message", {}).get("content") or c.get("text", "")
            else:
                msg = getattr(c, "message", None)
                text = getattr(msg, "content", "") if msg else getattr(c, "text", "")
            if text:
                samples.append(text)

        if len(samples) >= 2:
            # Extract optional context from input messages
            context = None
            messages = kwargs.get("messages", [])
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                if role in ("system", "developer"):
                    context = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    break

            receipt: AuditReceipt = self.guardrail.evaluate(
                sampled_responses=samples,
                context=context,
            )

            # Attach telemetry directly to the LiteLLM response object
            if hasattr(response_obj, "_spanda_receipt"):
                response_obj._spanda_receipt = receipt
            elif isinstance(response_obj, dict):
                response_obj["_spanda_receipt"] = receipt.to_dict()

            if self.block_mode and not receipt.is_safe:
                raise ValueError(
                    f"SpandaGuardrailBlocked: High uncertainty or mode collapse detected (R_sc={receipt.rsc:.3f}, decision={receipt.decision})"
                )
