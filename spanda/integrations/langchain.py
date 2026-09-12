"""
LangChain integration for Spanda ($R_{sc}$).

Provides:
- SpandaStringEvaluator: Custom evaluator matching LangChain StringEvaluator interface.
- SpandaGuardrailRunnable: LCEL Runnable for real-time hallucination & mode collapse filtering.
"""

from typing import List, Dict, Any, Optional, Sequence, Union
from spanda.core import compute_rsc
from spanda.guardrails import CascadedGuardrail, AuditReceipt

try:
    # Optional LangChain base classes
    from langchain_core.evaluators import StringEvaluator
    _HAS_LANGCHAIN = True
except ImportError:
    class StringEvaluator:
        """Fallback base class when langchain_core is not installed."""
        pass
    _HAS_LANGCHAIN = False


class SpandaStringEvaluator(StringEvaluator):
    """
    Evaluator for checking epistemic uncertainty and hallucination across sampled completions.
    
    Compatible with LangChain's evaluate_strings() contract.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        guardrail: Optional[CascadedGuardrail] = None
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.guardrail = guardrail or CascadedGuardrail(uncertainty_threshold=uncertainty_threshold)

    @property
    def evaluation_name(self) -> str:
        return "spanda_epistemic_uncertainty"

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    def evaluate_strings(
        self,
        *,
        prediction: Union[str, List[str]],
        input: Optional[str] = None,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate uncertainty on prediction or list of candidate predictions.
        
        Args:
            prediction: Single answer string or list of sampled generation paths.
            input: Optional prompt/question.
            reference: Optional ground-truth reference (if available).
            context: Optional retrieved context for RAG grounding check.
            
        Returns:
            Dict containing score (R_sc), value ("PASS" / "HALLUCINATION_RISK"),
            reasoning, and full AuditReceipt.
        """
        if isinstance(prediction, str):
            samples = [prediction]
        else:
            samples = list(prediction)

        receipt: AuditReceipt = self.guardrail.evaluate(
            samples=samples,
            context=context or reference
        )

        return {
            "key": self.evaluation_name,
            "score": round(receipt.rsc, 4),
            "value": "PASS" if receipt.is_safe else "HALLUCINATION_RISK",
            "decision": receipt.decision,
            "confidence": round(receipt.confidence, 4),
            "dominant_answer": receipt.dominant_answer,
            "tier_executed": receipt.tier_executed,
            "latency_ms": round(receipt.latency_ms, 3),
            "audit_receipt": receipt.to_dict()
        }


class SpandaGuardrailRunnable:
    """
    LangChain LCEL (LangChain Expression Language) compatible guardrail component.
    Wraps generation outputs and validates them against hallucinations.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        raise_on_hallucination: bool = False,
        guardrail: Optional[CascadedGuardrail] = None
    ):
        self.guardrail = guardrail or CascadedGuardrail(uncertainty_threshold=uncertainty_threshold)
        self.raise_on_hallucination = raise_on_hallucination

    def invoke(
        self,
        input_data: Union[List[str], Dict[str, Any]],
        config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes guardrail validation on candidate outputs.
        
        input_data can be:
        - A list of sampled generation strings: ["ans1", "ans2", ...]
        - A dict with {"samples": [...], "context": "..."}
        """
        if isinstance(input_data, list):
            samples = input_data
            context = None
        elif isinstance(input_data, dict):
            samples = input_data.get("samples", [])
            context = input_data.get("context")
        else:
            samples = [str(input_data)]
            context = None

        receipt = self.guardrail.evaluate(samples=samples, context=context)

        if not receipt.is_safe and self.raise_on_hallucination:
            raise ValueError(
                f"Spanda Guardrail Violation: decision={receipt.decision}, "
                f"R_sc={receipt.rsc:.4f} exceeds threshold."
            )

        return {
            "output": receipt.dominant_answer,
            "is_safe": receipt.is_safe,
            "receipt": receipt.to_dict(),
        }

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
