<div align="center">

# ⚡ Spanda ($R_{sc}$)
### Zero-Cost Epistemic Uncertainty Quantification for Large Language Models
[![PyPI](https://img.shields.io/pypi/v/spnda.svg?color=blue)](https://pypi.org/project/spnda/)
[![License: BSL 1.1 / MIT](https://img.shields.io/badge/License-BSL%201.1%20%2F%20MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22233648.svg)](https://doi.org/10.5281/zenodo.22233648)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--6038--1356-green.svg)](https://orcid.org/0009-0005-6038-1356)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

*Detect LLM hallucinations and quantify uncertainty in microseconds without secondary NLI cross-encoders.*

---

</div>

## 📌 Overview

Traditional epistemic uncertainty estimation in LLMs relies on **Semantic Entropy (SE)** ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664); [Farquhar et al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0)). While effective, Semantic Entropy requires clustering $K$ sampled generation paths using pairwise bidirectional NLI entailment classifiers (e.g., DeBERTa-v3-base). 

This introduces two severe production bottlenecks:
1. **Quadratic Cost:** $\binom{K}{2}$ forward passes per query (45 neural evaluations for $K=10$).
2. **Serving Latency:** Adds $\sim$90 ms of GPU overhead per inference call, making it unusable for high-throughput production serving.

**Spanda** introduces **Exact-Match Normalized Entropy ($R_{sc}$)**: a zero-parameter, zero-GPU metric that computes uncertainty directly over deterministic lexical clusters. 

Across empirical evaluations spanning two orders of magnitude (**1.5B to 120B parameters**), Spanda matches or exceeds neural Semantic Entropy on structured reasoning while operating **~90,000$\times$ faster** ($< 1\,\mu\text{s}$ vs. $92.4\,\text{ms}$).

---

## 🔬 Key Empirical Discoveries

### 1. The Coherence Scaling Law
As model capacity increases from 1.5B to 27B parameters, internal reasoning coherence causes correct predictions to naturally converge to identical lexical sequences. On mathematical reasoning (**GSM8K**), exact-match AUROC scales monotonically:

$$\text{AUROC}_{\text{GSM8K}}: \underbrace{0.577}_{\text{1.5B}} \longrightarrow \underbrace{0.706}_{\text{7B}} \longrightarrow \mathbf{\underbrace{0.889}_{\text{27B}}} \quad (p = 1.89 \times 10^{-28})$$

At **7B+ parameters**, Spanda achieves the exact same discriminative power as heavy DeBERTa-v3 NLI cross-encoders, rendering the neural clustering step redundant for reasoning.

### 2. Confident Mode Collapse (Safety Warning)
At the **120B frontier scale** on ungrounded factual recall (TriviaQA), the model exhibits **Confident Mode Collapse**: its parametric memory and RLHF tuning cause it to hallucinate the *exact same incorrect answer* identically across all $K$ paths. This yields an **inverted AUROC of 0.091** ($d = -2.23, p = 8.28 \times 10^{-15}$).

> ⚠️ **Critical Safety Implication:** Any system using self-consistency or agreement as a proxy for truth will be systematically deceived by frontier models on ungrounded factual recall. External grounding (RAG) is mandatory in this regime.

---

## 📊 Benchmark Results

| Model Scale | Benchmark | Accuracy | Spanda ($R_{sc}$) AUROC | Neural SE AUROC | Latency | GPU Req. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen-1.5B** | GSM8K | 11.4% | 0.577 | **0.584** | $<1\,\mu\text{s}$ | None |
| **Qwen-1.5B** | TriviaQA | 32.0% | 0.797 | **0.801** | $<1\,\mu\text{s}$ | None |
| **Mistral-7B** | GSM8K | 8.2% | **0.706** | 0.705 | $<1\,\mu\text{s}$ | None |
| **Mistral-7B** | TriviaQA | 45.0% | 0.698 | **0.755** | $<1\,\mu\text{s}$ | None |
| **Qwen-27B** | GSM8K | 61.2% | **0.889** | --- | $<1\,\mu\text{s}$ | None |
| **DeBERTa Baseline** | *N/A* | --- | --- | --- | **$\sim$92.4 ms** | Required |

### System Performance & Production Gateway Benchmarks

Empirical audit conducted across 50,000 evaluation iterations and 300 concurrent live HTTP reverse-proxy round-trips:

| Metric / Dimension | ⚡ Spanda Rust Gateway (`spnda`) | 🐢 LiteLLM Python (`litellm`) | Neural Semantic Entropy (DeBERTa) |
| :--- | :---: | :---: | :---: |
| **Mathematical Kernel Latency** | **652.1 nanoseconds (0.65 µs)** | ~15,000 µs (with neural judge) | 92,400 µs (92.4 ms) |
| **Kernel Throughput (Single Core)** | **1,533,500 evals/sec** | ~200,000 evals/sec (no-op hook) | ~10 evals/sec |
| **Cold Startup Time** | **3.69 ms** | **1,177.08 ms (1.17 s)** | N/A |
| **Memory Footprint (Idle RSS)** | **2.98 MB** | **229.61 MB** | ~1.8 GB GPU VRAM |
| **Proxy Net Latency Overhead** | **0.076 ms (76.3 µs)** | 12.0 – 28.0 ms (FastAPI/Uvicorn) | N/A |
| **Hardware Requirement** | **Pure CPU (Zero GPU)** | Pure CPU (plumbing) / GPU (judge) | Dedicated Nvidia GPU |

---

## 📐 Mathematical Formulation

Given $K$ sampled final answers $\{y_1, \dots, y_K\}$ for prompt $x$, deterministic normalization partitions them into $n$ equivalence classes $\{C_1, \dots, C_n\}$ with empirical probabilities $w_i = \frac{|C_i|}{K}$.

The **Normalized Shannon Entropy** is:
$$H_{\text{norm}} = \begin{cases} 0 & \text{if } n = 1 \\ \displaystyle\frac{-\sum_{i=1}^n w_i \ln w_i}{\ln K} & \text{if } n > 1 \end{cases}$$

The combined **Spanda Risk Score ($R_{sc}$)** balances entropy dispersion with modal dominance ($w_{\max} = \max_i w_i$):
$$R_{sc} = \alpha \cdot H_{\text{norm}} + (1 - \alpha) \cdot (1 - w_{\max}), \quad \alpha = 0.5$$

- $R_{sc} = 0$: Complete consensus (model is confident).
- $R_{sc} \to 1$: Maximum epistemic divergence (model is guessing / hallucinating).

---

## ⚡ Installation

Spanda is lightweight and requires **zero third-party dependencies** (pure Python standard library).

```bash
pip install spnda
```

*(Package name on PyPI is `spnda`; module is imported in Python as `import spanda`)*

Or install from source:

```bash
git clone https://github.com/Adarshent/Spnda.git
cd Spnda
pip install -e .
```

---

## 🚀 Quick Start

### 1. The 1-Line Client Wrapper (`spanda.wrap`)
Wrap any standard OpenAI, Groq, Ollama, or OpenAI-compatible client with transparent multi-path epistemic uncertainty quantification ($R_{sc}$), 7-state epistemic classification, and Confident Mode Collapse defense:

```python
import spanda
from openai import OpenAI

# 1-line drop-in wrapper (samples K=3 paths transparently)
client = spanda.wrap(OpenAI(), k=3, threshold=0.35, block=False)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 17 * 19?"}]
)

# Under the hood: runs in < 1 microsecond via native compiled Rust engine
print(response.spanda.rsc)             # 0.0000 (Unanimous consensus)
print(response.spanda.is_safe)         # True
print(response.spanda.decision)        # 'FAST_PASS_CONSISTENT'
print(response.spanda.latency_us)      # 0.7 µs!
print(response.choices[0].message.content) # Dominant consensus answer
```

If `block=True` is passed and the model hallucinates or diverges, `spanda.wrap` raises a `SpandaUncertaintyError` before invalid data reaches your users.

---

### 2. Standalone Rust Engine & OpenAI-Compatible Proxy (`spnda serve`)
For production microservices and non-Python languages (TypeScript, Go, Rust, Ruby, curl), run the standalone compiled Rust gateway:

```bash
# Launch the 760-nanosecond Rust proxy forwarding to any upstream LLM
spnda serve --upstream http://localhost:11434/v1 --port 8080 --block --k 3

# Or benchmark the mathematical engine directly
spnda bench --iterations 200000
# ✓ Latency per Eval : 767.9 nanoseconds
# ✓ Throughput       : 1,302,312 evaluations/sec on single core!

# Test any candidate completions via CLI
spnda eval "42" "42.0" "42"
```

Any application in any language can simply point `base_url="http://localhost:8080/v1"` to receive automatic sub-microsecond epistemic verification, Prometheus `/metrics`, and headers:
- `X-Spanda-Rsc: 0.0000`
- `X-Spanda-State: CONSISTENT`
- `X-Spanda-Decision: FAST_PASS_CONSISTENT`
- `X-Spanda-Latency-Us: 0.7`
- `X-Spanda-Attractor: false`

---

### 3. Core Epistemic Uncertainty & Hallucination API
```python
from spanda import compute_rsc, detect_hallucination, batch_compute_rsc

# 1. Basic Uncertainty Quantification
samples = ["Paris", "paris.", "Paris", "Paris", "Paris"]
res = compute_rsc(samples)
print(f"R_sc Score: {res['rsc']}")             # 0.0 (High confidence)
print(f"Dominant Answer: {res['dominant_answer']}") # 'Paris'

# 2. Production Hallucination Guardrail
guard = detect_hallucination(["42", "42", "24", "17", "99"], threshold=0.35)
if guard["is_uncertain"]:
    print(f"🚨 Hallucination Warning (R_sc = {guard['rsc']}). Routing to RAG / Review.")
else:
    print(f"✅ Safe output: {guard['dominant_answer']}")

# 3. High-Throughput Batch Processing
batch = [
    ["Answer A", "Answer A", "Answer A"],
    ["Choice 1", "Choice 2", "Choice 3"]
]
for r in batch_compute_rsc(batch):
    print(r["rsc"], r["dominant_answer"])
```

---

### 4. Enterprise Cascaded Guardrail (RAG & Autonomous Agents)

For mission-critical production pipelines, Spanda provides a **2-Tier Cascaded Guardrail** that combines sub-millisecond consensus filtering with context grounding and tool-call safety:

```python
from spanda import CascadedGuardrail

guard = CascadedGuardrail(
    uncertainty_threshold=0.3,
    grounding_threshold=0.15
)

# 1. RAG Query with Mode Collapse Protection
rag_context = "Documentation: The production cluster runs in us-east-1."
unanimous_hallucination = ["eu-west-3 Paris", "eu-west-3 Paris", "eu-west-3 Paris"]

receipt = guard.evaluate(unanimous_hallucination, context=rag_context)
print(receipt.decision)       # 'MODE_COLLAPSE_RISK'
print(receipt.is_safe)        # False (Unanimous agreement, but 0% grounded in source!)
print(receipt.tier_executed)  # Tier 2
print(receipt.latency_ms)     # < 0.05 ms

# 2. Agent Tool Call Argument Verification (e.g. preventing bad 'rm')
tool_calls = [
    {"command": "rm -rf /var/cache"},
    {"command": "rm -rf /var/log"},  # Conflict detected across parallel paths!
]
agent_receipt = guard.evaluate_tool_calls(tool_calls)
print(agent_receipt.decision) # 'TOOL_ARG_MISMATCH' (Execution blocked!)

# 3. Export SOC2 Audit Receipt
import json
print(json.dumps(receipt.to_dict(), indent=2))
```

---

### 5. Ecosystem Integrations (LangChain, LlamaIndex, LiteLLM)

Spanda connects into modern enterprise LLM pipelines with zero external dependencies:

```python
# 1. LangChain String Evaluator
from spanda.integrations.langchain import SpandaStringEvaluator

evaluator = SpandaStringEvaluator(uncertainty_threshold=0.35)
result = evaluator.evaluate_strings(
    prediction=["Paris", "Paris", "Paris", "Paris"],
    context="Paris is the capital of France."
)
print(result["value"])  # 'PASS' (Score: 0.0)

# 2. LlamaIndex Response Guardrail
from spanda.integrations.llamaindex import SpandaRAGGuardrail

guard = SpandaRAGGuardrail()
receipt = guard.validate_response(
    samples=["Result A", "Result A", "Result A"],
    context_str="Retrieved node knowledge..."
)
print(receipt.is_safe)  # True

# 3. LiteLLM Proxy / SDK Callback Hook
import litellm
from spanda.integrations.litellm import SpandaLiteLLMGuardrail

litellm.callbacks = [SpandaLiteLLMGuardrail(threshold=0.35, block_mode=False)]
```

---

### 6. Production Deployment & Observability

Run the standalone compiled Rust gateway in Docker or Kubernetes:

```bash
docker run -d -p 8080:8080 \
  -e SPANDA_UPSTREAM=https://api.openai.com/v1 \
  -e SPANDA_THRESHOLD=0.35 \
  spanda/spnda-gateway
```

#### Observability Endpoints:
- `GET /metrics`: Standard Prometheus format for Grafana (`spanda_requests_total`, `spanda_evaluations_total`, `spanda_mode_collapses_total`, `spanda_eval_latency_avg_us`).
- `GET /healthz`: Kubernetes liveness probe.
- `GET /readyz`: Kubernetes readiness probe.
- Structured JSON logging: Every transaction emits a machine-parseable log line to stdout for Datadog / CloudWatch / Splunk.

> ⚠️ **Operational Scope:** Spanda is engineered for **structured reasoning, math, code, agent tool-call arguments, SQL, and canonical factual RAG extraction** where 90ms GPU cross-encoders are an unacceptable bottleneck. It is **not** designed for open-ended, free-form creative prose (e.g., essays or poetry), where synonymous phrasing is naturally diverse and requires heavy neural NLI.

---

## 🛡️ Operational Envelope

| Use Case / Architecture | Recommendation | Rationale |
| :--- | :---: | :--- |
| **Math, Code & Structured QA (7B–70B)** | ✅ **Recommended** | Coherence Scaling Law ensures exact-match matches neural SE at 0 cost. |
| **High-Throughput Production APIs** | ✅ **Recommended** | 90,000x latency reduction without GPU requirements. |
| **Free-form Paraphrase QA (<7B)** | ⚠️ **Use Neural SE** | Small models produce inconsistent surface phrasing. |
| **Ungrounded Facts on Frontier Models (>100B)** | ❌ **Do Not Use Alone** | Subject to **Confident Mode Collapse**; must combine with retrieval (RAG). |

---

## 🧪 Testing

Run the test suite:

```bash
python3 -m unittest discover tests
```

---

## 📄 Citation

If you use Spanda in your research or production systems, please cite:

```bibtex
@article{nayak2026spanda,
  title={Spanda: Zero-Cost Lexical Entropy Matches Neural Semantic Uncertainty---Until Frontier Models Break It},
  author={Nayak, Bhupen},
  journal={arXiv preprint},
  year={2026},
  doi={10.5281/zenodo.22233648},
  url={https://doi.org/10.5281/zenodo.22233648}
}
```

---

## 📜 License & Governance

Spanda adopts a developer-friendly dual-licensing model:

- **Python SDK & Integrations (`spanda`)**: Permissive **[MIT License](LICENSE)**. Free for all developers, commercial and open-source applications, with zero dependency friction.
- **Compiled Rust Core Engine & Gateway (`spnda`)**: **[Business Source License 1.1 (BSL 1.1)](LICENSE)**. Free for developers, research, and internal production infrastructure. Prohibits offering Spanda as a competing commercial third-party managed service without an enterprise license from Spanda Research. Automatically converts to Apache 2.0 on January 1, 2030.
