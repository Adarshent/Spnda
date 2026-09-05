"""
Enterprise Cascaded Guardrail Demo
Shows how to use Spanda's 2-Tier Cascaded Guardrail in real-world RAG & Agent pipelines.
"""

import os
import sys
import json

# Ensure local spanda is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spanda import CascadedGuardrail

def main():
    print("=" * 70)
    print("SPANDA ENTERPRISE CASCADED GUARDRAIL DEMO")
    print("=" * 70)

    guard = CascadedGuardrail(
        uncertainty_threshold=0.3,
        grounding_threshold=0.2
    )

    # -------------------------------------------------------------
    # Scenario 1: High Entropy / Conflicting Paths -> Tier 1 Early Exit
    # -------------------------------------------------------------
    print("\n[Scenario 1: High Entropy Generation -> Tier 1 Early Exit]")
    conflicting_answers = [
        "The server port is 8080",
        "The server port is 443",
        "The server port is 3000",
        "The server port is 8000"
    ]
    receipt1 = guard.evaluate(conflicting_answers)
    print(f"Decision: {receipt1.decision}")
    print(f"Is Safe: {receipt1.is_safe}")
    print(f"Tier Executed: Tier {receipt1.tier_executed}")
    print(f"Latency: {receipt1.latency_ms:.3f} ms")
    print(f"Action: {receipt1.details.get('action')}")

    # -------------------------------------------------------------
    # Scenario 2: Grounded Safe RAG Response -> Tier 2 Pass
    # -------------------------------------------------------------
    print("\n[Scenario 2: Grounded RAG Query -> Tier 2 Verified Safe]")
    context = (
        "Documentation: The primary database failover cluster is located in us-east-1. "
        "The replica endpoint runs on db.production.internal."
    )
    rag_answers = [
        "us-east-1",
        "us-east-1",
        "us-east-1",
        "us-east-1"
    ]
    receipt2 = guard.evaluate(rag_answers, context=context)
    print(f"Decision: {receipt2.decision}")
    print(f"Is Safe: {receipt2.is_safe}")
    print(f"Grounding Overlap: {receipt2.grounding_overlap * 100:.1f}%")
    print(f"Tier Executed: Tier {receipt2.tier_executed}")
    print(f"Latency: {receipt2.latency_ms:.3f} ms")

    # -------------------------------------------------------------
    # Scenario 3: Confident Mode Collapse -> Tier 2 Alert!
    # (Model outputs unanimous wrong answer with ZERO context grounding)
    # -------------------------------------------------------------
    print("\n[Scenario 3: Confident Mode Collapse (Unanimous Hallucination)]")
    hallucinated_answers = [
        "eu-west-3 Paris",
        "eu-west-3 Paris",
        "eu-west-3 Paris",
        "eu-west-3 Paris"
    ]
    receipt3 = guard.evaluate(hallucinated_answers, context=context)
    print(f"Decision: {receipt3.decision}")
    print(f"Is Safe: {receipt3.is_safe}")
    print(f"R_sc Score: {receipt3.rsc} (100% Unanimous!)")
    print(f"Grounding Overlap: {receipt3.grounding_overlap * 100:.1f}%")
    print(f"Alert: {receipt3.details.get('warning')}")

    # -------------------------------------------------------------
    # Scenario 4: Coding Agent Tool-Call Argument Verification
    # (Ensures parallel agent paths agree on tool arguments before execution)
    # -------------------------------------------------------------
    print("\n[Scenario 4: Agent Tool Call Argument Guardrail]")
    conflicting_tool_calls = [
        {"tool": "bash", "command": "rm -rf /var/cache/apt"},
        {"tool": "bash", "command": "rm -rf /var/log/syslog"},
        {"tool": "bash", "command": "rm -rf /tmp/data"},
    ]
    receipt4 = guard.evaluate_tool_calls(conflicting_tool_calls)
    print(f"Decision: {receipt4.decision}")
    print(f"Is Safe: {receipt4.is_safe}")
    print(f"Action: {receipt4.details.get('action')}")
    print(f"Warning: {receipt4.details.get('warning')}")

    # -------------------------------------------------------------
    # Scenario 5: SOC2 / Audit Receipt JSON Export
    # -------------------------------------------------------------
    print("\n[Scenario 5: Audit Receipt Telemetry Export]")
    print(json.dumps(receipt2.to_dict(), indent=2))

if __name__ == "__main__":
    main()
