"""
Spanda — Sub-Microsecond Epistemic Uncertainty & High-Throughput LLM Gateway.

Detect hallucinations and confident mode collapse in LLM outputs using
exact-match discrete equivalence partitioning and normalized entropy (R_sc).
Over 90,000x faster than neural Semantic Entropy, zero GPU required.

Paper: "Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty
        — Until Frontier Models Break It" (Nayak, 2026)
Zenodo DOI: 10.5281/zenodo.22233648

Usage:
    >>> import spanda
    >>> from openai import OpenAI
    >>> client = spanda.wrap(OpenAI(), k=3, threshold=0.35)
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "What is 17 * 19?"}]
    ... )
    >>> print(response.spanda.rsc)        # 0.0
    >>> print(response.spanda.is_safe)    # True
"""

__version__ = "0.3.0"
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
from spanda.wrapper import (
    wrap,
    SpandaUncertaintyError,
    SpandaEvaluation,
)
from spanda.integrations import (
    SpandaStringEvaluator,
    SpandaGuardrailRunnable,
    SpandaRAGGuardrail,
    SpandaLiteLLMGuardrail,
)
from spanda.gateway import run_gateway, GatewayMetrics

__all__ = [
    "wrap",
    "SpandaUncertaintyError",
    "SpandaEvaluation",
    "compute_rsc",
    "normalize_answer",
    "detect_hallucination",
    "batch_compute_rsc",
    "CascadedGuardrail",
    "AuditReceipt",
    "SpandaStringEvaluator",
    "SpandaGuardrailRunnable",
    "SpandaRAGGuardrail",
    "SpandaLiteLLMGuardrail",
    "run_gateway",
    "GatewayMetrics",
]
