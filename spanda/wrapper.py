"""
Spanda Universal 1-Line Client Wrapper (Spanda Research).

Wraps any OpenAI, Groq, Ollama, Together, or OpenAI-compatible client
with automatic multi-path epistemic uncertainty quantification ($R_{sc}$),
7-state epistemic classification, and Confident Mode Collapse defense.

Usage:
    >>> import spanda
    >>> from openai import OpenAI
    >>> client = spanda.wrap(OpenAI(), k=3, threshold=0.35, block=False)
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o-mini",
    ...     messages=[{"role": "user", "content": "What is 17 * 19?"}]
    ... )
    >>> print(response.spanda.rsc)        # 0.0
    >>> print(response.spanda.is_safe)    # True
    >>> print(response.spanda.decision)   # FAST_PASS_CONSISTENT
"""

import os
import sys
import json
import ctypes
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Dict, Union

from spanda.core import compute_rsc
from spanda.guardrails import CascadedGuardrail, AuditReceipt


class SpandaUncertaintyError(Exception):
    """Raised when an LLM completion exceeds the epistemic uncertainty threshold in block mode."""
    def __init__(self, message: str, receipt: Any):
        super().__init__(message)
        self.receipt = receipt


class SpandaEvaluation:
    """Structured audit container attached to every guarded response."""
    def __init__(
        self,
        rsc: float,
        is_safe: bool,
        decision: str,
        dominant_answer: str,
        k_samples: int,
        agreement_ratio: float,
        mode_collapse: bool,
        latency_us: float,
        raw_receipt: Any = None,
    ):
        self.rsc = rsc
        self.is_safe = is_safe
        self.decision = decision
        self.dominant_answer = dominant_answer
        self.k_samples = k_samples
        self.agreement_ratio = agreement_ratio
        self.mode_collapse = mode_collapse
        self.latency_us = latency_us
        self.raw_receipt = raw_receipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsc": self.rsc,
            "is_safe": self.is_safe,
            "decision": self.decision,
            "dominant_answer": self.dominant_answer,
            "k_samples": self.k_samples,
            "agreement_ratio": self.agreement_ratio,
            "mode_collapse": self.mode_collapse,
            "latency_us": self.latency_us,
        }

    def __repr__(self) -> str:
        return (
            f"SpandaEvaluation(rsc={self.rsc:.4f}, safe={self.is_safe}, "
            f"decision='{self.decision}', dominant='{self.dominant_answer[:30]}...', "
            f"latency={self.latency_us:.1f}µs)"
        )


# Try to locate compiled Rust cdylib for sub-microsecond C-FFI execution
_RUST_LIB = None
try:
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dylib_patterns = [
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "release", "*spanda_core*.dylib"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "release", "*spanda_core*.so"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "debug", "*spanda_core*.dylib"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "debug", "*spanda_core*.so"),
    ]
    for pattern in dylib_patterns:
        matches = glob.glob(pattern)
        if matches:
            lib = ctypes.CDLL(matches[0])
            lib.spanda_eval_c.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
            lib.spanda_eval_c.restype = ctypes.c_void_p
            lib.spanda_free_string.argtypes = [ctypes.c_void_p]
            lib.spanda_free_string.restype = None
            _RUST_LIB = lib
            break
except Exception:
    _RUST_LIB = None


def _evaluate_fast(samples: List[str], context: Optional[str] = None, threshold: float = 0.35) -> SpandaEvaluation:
    """Evaluates samples via compiled Rust C-ABI engine if available, falling back to Python."""
    global _RUST_LIB
    if _RUST_LIB is not None:
        try:
            samples_json = json.dumps(samples).encode("utf-8")
            ctx_bytes = context.encode("utf-8") if context else None
            ptr = _RUST_LIB.spanda_eval_c(samples_json, ctx_bytes, ctypes.c_double(threshold))
            if ptr:
                res_str = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
                _RUST_LIB.spanda_free_string(ptr)
                data = json.loads(res_str)
                return SpandaEvaluation(
                    rsc=data.get("rsc", 0.0),
                    is_safe=data.get("is_safe", True),
                    decision=data.get("decision", "PASS"),
                    dominant_answer=data.get("dominant_answer", ""),
                    k_samples=data.get("k_samples", len(samples)),
                    agreement_ratio=data.get("agreement_ratio", 1.0),
                    mode_collapse=data.get("attractor_collapse", False),
                    latency_us=data.get("latency_micros", 1.2),
                    raw_receipt=data,
                )
        except Exception:
            pass

    # Pure Python zero-dependency fallback
    guard = CascadedGuardrail(uncertainty_threshold=threshold)
    receipt: AuditReceipt = guard.evaluate(sampled_responses=samples, context=context)
    return SpandaEvaluation(
        rsc=receipt.rsc,
        is_safe=receipt.is_safe,
        decision=receipt.decision,
        dominant_answer=receipt.dominant_answer,
        k_samples=receipt.k_samples,
        agreement_ratio=receipt.agreement_ratio,
        mode_collapse=bool(receipt.details.get("mode_collapse_flag", False)),
        latency_us=receipt.latency_ms * 1000.0,
        raw_receipt=receipt,
    )


def wrap(
    client: Any,
    k: int = 3,
    threshold: float = 0.35,
    block: bool = False,
) -> Any:
    """
    Wraps an OpenAI-compatible client with transparent epistemic uncertainty quantification.

    Args:
        client: The initialized LLM client (e.g. OpenAI(), Groq())
        k: Number of paths to sample for uncertainty quantification (default: 3)
        threshold: Normalized entropy R_sc cutoff threshold (default: 0.35)
        block: If True, raises SpandaUncertaintyError when output fails safety check.
    """
    orig_completions = getattr(client.chat, "completions", None)
    if orig_completions is None:
        raise AttributeError("Client does not have a standard chat.completions interface")

    orig_create = orig_completions.create

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # Pass streaming requests through
        if kwargs.get("stream", False):
            return orig_create(*args, **kwargs)

        orig_n = kwargs.get("n", 1)
        messages = kwargs.get("messages", [])

        # Extract context
        context = None
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            if role in ("system", "developer"):
                context = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                break

        # Attempt to request n=k directly from provider
        responses = []
        samples = []
        target_k = max(k, orig_n)

        try:
            req_kwargs = dict(kwargs)
            req_kwargs["n"] = target_k
            resp = orig_create(*args, **req_kwargs)
            responses = [resp]
            for c in getattr(resp, "choices", []):
                msg = getattr(c, "message", None)
                text = getattr(msg, "content", "") if msg else getattr(c, "text", "")
                if text:
                    samples.append(text)
        except Exception:
            # Fallback for providers that don't support n > 1 (e.g., Anthropic, certain local Ollama models)
            # Query K paths concurrently using ThreadPoolExecutor
            def _single_query():
                req_kw = dict(kwargs)
                req_kw["n"] = 1
                return orig_create(*args, **req_kw)

            with ThreadPoolExecutor(max_workers=target_k) as executor:
                futures = [executor.submit(_single_query) for _ in range(target_k)]
                for f in futures:
                    try:
                        r = f.result()
                        responses.append(r)
                        choices = getattr(r, "choices", [])
                        if choices:
                            msg = getattr(choices[0], "message", None)
                            text = getattr(msg, "content", "") if msg else getattr(choices[0], "text", "")
                            if text:
                                samples.append(text)
                    except Exception:
                        pass

        if not samples:
            # If sampling failed completely, execute normal single call
            return orig_create(*args, **kwargs)

        eval_result = _evaluate_fast(samples, context=context, threshold=threshold)

        # Base response object to return
        main_resp = responses[0]
        setattr(main_resp, "spanda", eval_result)

        # If original call requested n=1, select dominant answer
        if orig_n == 1:
            choices = getattr(main_resp, "choices", [])
            if len(choices) > 1:
                dominant = eval_result.dominant_answer.strip()
                best_choice = choices[0]
                for c in choices:
                    msg = getattr(c, "message", None)
                    text = (getattr(msg, "content", "") if msg else getattr(c, "text", "")).strip()
                    if text == dominant:
                        best_choice = c
                        break
                main_resp.choices = [best_choice]

        if block and not eval_result.is_safe:
            raise SpandaUncertaintyError(
                f"Spanda guardrail blocked completion: R_sc={eval_result.rsc:.3f} "
                f"(decision='{eval_result.decision}', safe=False)",
                receipt=eval_result,
            )

        return main_resp

    client.chat.completions.create = wrapped_create
    return client
