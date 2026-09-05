"""
Spanda — Zero-Cost Epistemic Uncertainty & Cascaded Guardrails for LLMs.

Detect hallucinations in LLM outputs using exact-match normalized entropy.
90,000x faster than neural Semantic Entropy, zero GPU required.

Paper: "Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty
        — Until Frontier Models Break It" (Nayak, 2026)

Usage:
    >>> from spanda import compute_rsc, detect_hallucination, CascadedGuardrail
    >>> answers = ["42", "42", "42", "41", "43"]
    >>> result = compute_rsc(answers)
    >>> print(result)

    >>> # Enterprise Cascaded Guardrail
    >>> guard = CascadedGuardrail(uncertainty_threshold=0.3)
    >>> receipt = guard.evaluate(answers, context="The answer is forty-two.")
    >>> print(receipt)
"""

__version__ = "0.2.0"
__author__ = "Bhupen Nayak"
__email__ = "bhupennayak@icloud.com"

from spanda.core import (
    compute_rsc,
    normalize_answer,
    detect_hallucination,
    batch_compute_rsc,
)
from spanda.guardrails import (
    CascadedGuardrail,
    AuditReceipt,
)

__all__ = [
    "compute_rsc",
    "normalize_answer",
    "detect_hallucination",
    "batch_compute_rsc",
    "CascadedGuardrail",
    "AuditReceipt",
]
