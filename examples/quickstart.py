"""
Spanda Quickstart Example
-------------------------
Demonstrating how to use Spanda (R_sc) to detect hallucinations 
and quantify epistemic uncertainty from sampled LLM responses.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spanda import compute_rsc, detect_hallucination, batch_compute_rsc

def main():
    print("=" * 60)
    print("Spanda: Zero-Cost Epistemic Uncertainty Estimation for LLMs")
    print("=" * 60)

    # -------------------------------------------------------------
    # Scenario 1: Model has high confidence / internal consensus
    # Prompt: "What is the capital of France?"
    # -------------------------------------------------------------
    confident_samples = [
        "Paris",
        "Paris.",
        "paris",
        "Paris",
        "Paris"
    ]
    res1 = compute_rsc(confident_samples)
    print("\n[Scenario 1] High Consensus Query:")
    print(f"  Samples: {confident_samples}")
    print(f"  Dominant Answer: {res1['dominant_answer']}")
    print(f"  Risk Score (R_sc): {res1['rsc']} (Low risk, consensus reached)")
    print(f"  Modal Dominance (w_max): {res1['w_max']}")
    print(f"  Normalized Entropy: {res1['h_norm']}")

    # -------------------------------------------------------------
    # Scenario 2: Model is guessing / hallucinating
    # Prompt: "What was the population of Atlantis in 300 BC?"
    # -------------------------------------------------------------
    uncertain_samples = [
        "50,000",
        "100,000",
        "25,000",
        "over 1 million",
        "unknown"
    ]
    res2 = compute_rsc(uncertain_samples)
    print("\n[Scenario 2] High Uncertainty / Guessing Query:")
    print(f"  Samples: {uncertain_samples}")
    print(f"  Risk Score (R_sc): {res2['rsc']} (High risk, epistemic divergence)")
    print(f"  Clusters Found: {res2['n_clusters']}")

    # -------------------------------------------------------------
    # Scenario 3: Guardrail Filter / Routing
    # -------------------------------------------------------------
    guard = detect_hallucination(uncertain_samples, threshold=0.35)
    print("\n[Scenario 3] Production Guardrail Filter:")
    if guard["is_uncertain"]:
        print(f"  🚨 Triggered Guardrail (R_sc = {guard['rsc']} > 0.35)!")
        print("  Action: Route query to Retrieval-Augmented Generation (RAG) or human review.")
    else:
        print(f"  ✅ Safe to return answer: {guard['dominant_answer']}")

    # -------------------------------------------------------------
    # Scenario 4: Batch Evaluation
    # -------------------------------------------------------------
    batch = [confident_samples, uncertain_samples]
    batch_results = batch_compute_rsc(batch)
    print(f"\n[Scenario 4] Batch Evaluation Processed: {len(batch_results)} items.")

if __name__ == "__main__":
    main()
