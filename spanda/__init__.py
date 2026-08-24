"""
Spanda — Zero-Cost Epistemic Uncertainty for LLMs.

Detect hallucinations in LLM outputs using exact-match normalized entropy.
90,000x faster than neural Semantic Entropy, zero GPU required.

Paper: "Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty
        — Until Frontier Models Break It" (Nayak, 2026)

Usage:
    >>> from spanda import compute_rsc, detect_hallucination
    >>> answers = ["42", "42", "42", "41", "43"]
    >>> result = compute_rsc(answers)
    >>> print(result)
"""

__version__ = "0.1.0"
__author__ = "Bhupen Nayak"
__email__ = "bhupennayak@icloud.com"

from spanda.core import (
    compute_rsc,
    normalize_answer,
    detect_hallucination,
    batch_compute_rsc,
)

__all__ = [
    "compute_rsc",
    "normalize_answer",
    "detect_hallucination",
    "batch_compute_rsc",
]
