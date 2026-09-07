"""
BRHMN Spanda Gateway Demonstration.

Shows how to use Spanda Gateway as a drop-in reverse proxy for OpenAI / Groq / vLLM.
Injects sub-millisecond epistemic uncertainty and cascaded guardrail headers.

Run Gateway:
    python -m spanda.gateway --upstream https://api.openai.com/v1 --port 8080

Client Code (Standard OpenAI SDK):
"""

import os
# Example client interaction (requires openai package if executed against live LLM)
def demo_client():
    print("=" * 70)
    print("BRHMN SPANDA GATEWAY — DROP-IN REVERSE PROXY DEMO")
    print("=" * 70)
    print("\n1. How developers point their existing code:")
    print("""
    from openai import OpenAI

    # Simply change base_url to the local Spanda Gateway:
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key=os.environ.get("OPENAI_API_KEY", "mock-key")
    )

    # Every call is automatically guarded in <0.05ms:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        # Spanda Gateway intercepts, samples K paths, and evaluates R_sc:
    )
    """)

    print("2. Response Telemetry Injected by Gateway:")
    print("   → X-Spanda-Rsc: 0.0000          (Consensus reached)")
    print("   → X-Spanda-Safe: true           (Verified grounded)")
    print("   → X-Spanda-Decision: VERIFIED_SAFE_GROUNDED")
    print("   → X-Spanda-Latency-Ms: 0.042 ms (Sub-millisecond)")
    print("   → X-Spanda-Tier: 1              (Fast-path consensus exit)")
    print("\nIf the model hallucinates (R_sc > 0.35):")
    print("   → X-Spanda-Safe: false")
    print("   → X-Spanda-Decision: UNCERTAIN_HIGH_ENTROPY")
    print("   → HTTP 422 Rejection (if --block mode is enabled)")
    print("=" * 70)

if __name__ == "__main__":
    demo_client()
