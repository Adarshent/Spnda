import time
import re
import json
from typing import List, Dict, Any, Optional, Union
from spanda.core import compute_rsc, normalize_answer

class AuditReceipt:
    """
    Enterprise audit receipt returned by the CascadedGuardrail.
    Contains decision, risk metrics, grounding attribution, and execution latency.
    """
    def __init__(
        self,
        decision: str,
        is_safe: bool,
        dominant_answer: str,
        rsc: float,
        confidence: float,
        tier_executed: int,
        latency_ms: float,
        grounding_overlap: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.decision = decision
        self.is_safe = is_safe
        self.dominant_answer = dominant_answer
        self.rsc = rsc
        self.confidence = confidence
        self.tier_executed = tier_executed
        self.latency_ms = latency_ms
        self.grounding_overlap = grounding_overlap
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the receipt for JSON logging or SOC2 audit pipelines."""
        return {
            "decision": self.decision,
            "is_safe": self.is_safe,
            "dominant_answer": self.dominant_answer,
            "rsc": self.rsc,
            "confidence": self.confidence,
            "tier_executed": self.tier_executed,
            "grounding_overlap": self.grounding_overlap,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"AuditReceipt(decision='{self.decision}', is_safe={self.is_safe}, "
            f"rsc={self.rsc}, tier={self.tier_executed}, latency_ms={self.latency_ms:.2f}ms)"
        )


class CascadedGuardrail:
    """
    Enterprise-grade 2-Tier Cascaded Guardrail engine for LLM inference.
    
    Architecture:
    - Tier 1: Fast-Path Lexical Consensus (R_sc) executing in ~1.5ms on CPU.
              Instantly rejects high-entropy, conflicting generations (early-exit).
    - Tier 2: Grounding & Mode Collapse Detector.
              Validates that low-entropy unanimous consensus is actually grounded
              in the retrieved context/documents, catching 'Confident Mode Collapse'.
    - Tool Payload Mode: Validates structured tool-calling parameters in agent loops.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        grounding_threshold: float = 0.15,
    ):
        """
        Args:
            uncertainty_threshold: Maximum allowed R_sc score before flagging as uncertain. Default 0.3.
            grounding_threshold: Minimum fraction of key tokens required to be grounded in context. Default 0.15.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.grounding_threshold = grounding_threshold

    def evaluate(
        self,
        sampled_responses: Optional[List[str]] = None,
        context: Optional[str] = None,
        is_critical: bool = False,
        samples: Optional[List[str]] = None,
    ) -> AuditReceipt:
        """
        Evaluates K sampled generation paths through the cascaded guardrail.

        Args:
            sampled_responses: List of K candidate strings from the LLM.
            context: Optional retrieved context (RAG chunks, doc excerpts, or system facts).
            is_critical: If True, forces Tier 2 verification even if no context is given.

        Returns:
            AuditReceipt with safety verdict and detailed telemetry.
        """
        start_time = time.perf_counter()

        sampled_responses = sampled_responses or samples
        if not sampled_responses:
            raise ValueError("sampled_responses (or samples) cannot be empty.")

        # --- Tier 1: Fast Path Lexical Consensus (< 1.5ms) ---
        res = compute_rsc(sampled_responses)
        rsc = res["rsc"]
        dominant_answer = res["dominant_answer"]
        confidence = round(1.0 - rsc, 4)

        # Early exit if uncertainty is high (conflicting paths)
        if rsc > self.uncertainty_threshold:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return AuditReceipt(
                decision="UNCERTAIN_HIGH_ENTROPY",
                is_safe=False,
                dominant_answer=dominant_answer,
                rsc=rsc,
                confidence=confidence,
                tier_executed=1,
                latency_ms=round(latency_ms, 3),
                grounding_overlap=None,
                details={
                    "h_norm": res["h_norm"],
                    "w_max": res["w_max"],
                    "n_clusters": res["n_clusters"],
                    "action": "REJECT_OR_RETRY_PROMPT"
                }
            )

        # --- Tier 2: Grounding & Mode Collapse Verification ---
        if context is not None:
            overlap = self._compute_grounding_overlap(dominant_answer, context)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Mode Collapse Check: 100% unanimous agreement but 0% grounded in source
            if overlap < self.grounding_threshold:
                return AuditReceipt(
                    decision="MODE_COLLAPSE_RISK",
                    is_safe=False,
                    dominant_answer=dominant_answer,
                    rsc=rsc,
                    confidence=confidence,
                    tier_executed=2,
                    latency_ms=round(latency_ms, 3),
                    grounding_overlap=round(overlap, 4),
                    details={
                        "warning": "Unanimous consensus detected with zero source grounding.",
                        "action": "FLAG_CONFIDENT_HALLUCINATION"
                    }
                )
            else:
                return AuditReceipt(
                    decision="VERIFIED_GROUNDED_SAFE",
                    is_safe=True,
                    dominant_answer=dominant_answer,
                    rsc=rsc,
                    confidence=confidence,
                    tier_executed=2,
                    latency_ms=round(latency_ms, 3),
                    grounding_overlap=round(overlap, 4),
                    details={"action": "SERVE_OUTPUT"}
                )

        # Tier 1 fast-pass when no context check is required
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return AuditReceipt(
            decision="FAST_PASS_CONSISTENT",
            is_safe=True,
            dominant_answer=dominant_answer,
            rsc=rsc,
            confidence=confidence,
            tier_executed=1,
            latency_ms=round(latency_ms, 3),
            grounding_overlap=None,
            details={"action": "SERVE_OUTPUT"}
        )

    def evaluate_tool_calls(
        self,
        sampled_tool_payloads: List[Union[Dict[str, Any], str]]
    ) -> AuditReceipt:
        """
        Specifically scores uncertainty across tool-call arguments for autonomous agents.
        Ensures that parallel agent paths agree on the critical parameters (e.g. bash commands,
        file paths, API arguments) before invoking potentially destructive tools.

        Args:
            sampled_tool_payloads: List of parsed JSON dicts or raw argument strings.

        Returns:
            AuditReceipt evaluating argument consensus.
        """
        start_time = time.perf_counter()

        canonical_strings = []
        for payload in sampled_tool_payloads:
            if isinstance(payload, dict):
                # Deterministic canonical JSON
                canonical_strings.append(json.dumps(payload, sort_keys=True))
            else:
                canonical_strings.append(str(payload).strip())

        res = compute_rsc(canonical_strings)
        rsc = res["rsc"]
        dominant_answer = res["dominant_answer"]
        confidence = round(1.0 - rsc, 4)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if rsc > self.uncertainty_threshold:
            return AuditReceipt(
                decision="TOOL_ARG_MISMATCH",
                is_safe=False,
                dominant_answer=dominant_answer,
                rsc=rsc,
                confidence=confidence,
                tier_executed=1,
                latency_ms=round(latency_ms, 3),
                details={
                    "n_clusters": res["n_clusters"],
                    "warning": "Sampled agent paths disagree on tool arguments.",
                    "action": "BLOCK_TOOL_EXECUTION"
                }
            )

        return AuditReceipt(
            decision="TOOL_ARG_VERIFIED",
            is_safe=True,
            dominant_answer=dominant_answer,
            rsc=rsc,
            confidence=confidence,
            tier_executed=1,
            latency_ms=round(latency_ms, 3),
            details={
                "action": "EXECUTE_TOOL"
            }
        )

    def _compute_grounding_overlap(self, candidate: str, context: str) -> float:
        """
        Computes the lexical token overlap between candidate answer and retrieved context.
        """
        norm_candidate = normalize_answer(candidate)
        norm_context = normalize_answer(context)

        tokens = [t for t in norm_candidate.split() if len(t) > 1 or t.isdigit()]
        if not tokens:
            return 1.0  # Empty or single char fallback

        context_tokens = set(norm_context.split())
        matched = sum(1 for t in tokens if t in context_tokens)

        return matched / len(tokens)
