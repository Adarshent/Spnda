# Spanda ($R_{sc}$)

> **Zero-Cost Epistemic Uncertainty Estimation for Large Language Models**

Spanda is a Python package for detecting LLM hallucinations using exact-match normalized entropy ($R_{sc}$). It is designed to be a **zero-parameter, zero-latency** alternative to neural Semantic Entropy. 

For reasoning tasks on 7B-27B models, Spanda matches or exceeds the AUROC of $O(K^2)$ DeBERTa-v3 cross-encoders while running **90,000$\times$ faster** (microseconds instead of milliseconds) and requiring zero GPUs.

📖 **Paper:** [Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty—Until Frontier Models Break It](https://arxiv.org/abs/2608.xxxxx)

## 🚀 Installation

```bash
pip install spanda
```

## 💻 Quick Start

To quantify uncertainty, simply sample $K$ multiple paths from your LLM (e.g., at temperature $T=0.7$) and pass them to Spanda.

```python
from spanda import compute_rsc, detect_hallucination

# 1. The model is highly confident (High consensus)
confident_answers = ["Paris", "paris.", "Paris", "Paris", "Paris"]
result = compute_rsc(confident_answers)
print(f"Confident R_sc: {result['rsc']}")  # Output: 0.0


# 2. The model is guessing/hallucinating (High entropy)
uncertain_answers = ["Berlin", "Madrid", "Rome", "London", "Paris"]
result = compute_rsc(uncertain_answers)
print(f"Uncertain R_sc: {result['rsc']}")  # Output: 0.9


# 3. Use the convenience detector
is_hallucinating = detect_hallucination(uncertain_answers, threshold=0.3)
if is_hallucinating['is_uncertain']:
    print("Warning: Model is likely hallucinating. Route to human or RAG.")
```

## ⚠️ The Operational Envelope (Safety Warning)

Please read the paper before deploying this in production. Spanda establishes a **Coherence Scaling Law**, showing that $R_{sc}$ becomes highly reliable for math and reasoning on modern 7B and 27B models.

However, our research also discovered **Confident Mode Collapse**: On frontier models (>100B parameters) executing factual recall, the model's RLHF tuning causes it to hallucinate the *exact same incorrect answer* 100% of the time, bypassing self-consistency checks. 

**Rule of Thumb:**
- ✅ **DO USE** Spanda for math, code, and structured QA on 7B to 70B models.
- ❌ **DO NOT USE** Spanda (or any self-consistency method) to verify ungrounded trivia/facts on frontier models like GPT-4 or 120B+ architectures without external RAG.

## Citation

If you use Spanda in your research, please cite our paper:

```bibtex
@article{nayak2026spanda,
  title={Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty---Until Frontier Models Break It},
  author={Nayak, Bhupen},
  journal={arXiv preprint arXiv:2608.xxxxx},
  year={2026}
}
```
