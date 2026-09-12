"""
LlamaIndex integration for Spanda ($R_{sc}$).

Provides:
- SpandaRAGGuardrail: Node post-processor and response validator for LlamaIndex RAG query engines.
"""

from typing import List, Dict, Any, Optional, Union
from spanda.guardrails import CascadedGuardrail, AuditReceipt

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    _HAS_LLAMAINDEX = True
except ImportError:
    class BaseNodePostprocessor:
        """Fallback when llama-index is not installed."""
        pass
    _HAS_LLAMAINDEX = False


class SpandaRAGGuardrail(BaseNodePostprocessor):
    """
    LlamaIndex post-processor and response guardrail.
    Validates sampled LLM responses against retrieved context nodes.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        guardrail: Optional[CascadedGuardrail] = None
    ):
        self.guardrail = guardrail or CascadedGuardrail(uncertainty_threshold=uncertainty_threshold)

    def validate_response(
        self,
        samples: List[str],
        retrieved_nodes: Optional[List[Any]] = None,
        context_str: Optional[str] = None
    ) -> AuditReceipt:
        """
        Validates LLM response samples against retrieved RAG context.
        
        Args:
            samples: List of generated answers.
            retrieved_nodes: List of LlamaIndex NodeWithScore objects.
            context_str: Optional pre-joined context string.
            
        Returns:
            AuditReceipt from CascadedGuardrail.
        """
        if context_str is None and retrieved_nodes:
            # Extract text from LlamaIndex nodes
            texts = []
            for node in retrieved_nodes:
                if hasattr(node, "node") and hasattr(node.node, "get_content"):
                    texts.append(node.node.get_content())
                elif hasattr(node, "get_content"):
                    texts.append(node.get_content())
                elif hasattr(node, "text"):
                    texts.append(node.text)
                else:
                    texts.append(str(node))
            context_str = " ".join(texts)

        return self.guardrail.evaluate(samples=samples, context=context_str)

    def postprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None
    ) -> List[Any]:
        """
        BaseNodePostprocessor hook for LlamaIndex query engines.
        Returns nodes as-is while enabling guardrail pipeline attachment.
        """
        return nodes
