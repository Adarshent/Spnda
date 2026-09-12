"""
Spanda Unified CLI Interface (Spanda Research).

Routes commands to native Rust high-performance engine if compiled,
or pure-Python standard library gateway fallback.
"""

import os
import sys
import glob
import subprocess
import argparse

from spanda.gateway import run_gateway
from spanda.core import compute_rsc
from spanda.guardrails import CascadedGuardrail


def _find_native_binary():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "release", "spnda"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "release", "spanda"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "debug", "spnda"),
        os.path.join(pkg_dir, "crates", "spanda-core", "target", "debug", "spanda"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def main():
    # If compiled Rust binary exists, transparently forward all arguments to it!
    native_bin = _find_native_binary()
    if native_bin and len(sys.argv) > 1 and sys.argv[1] != "--python":
        try:
            os.execv(native_bin, [native_bin] + sys.argv[1:])
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="spnda",
        description="Spanda — Epistemic Uncertainty Quantification & Guardrail Gateway (Spanda Research)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Launch OpenAI-compatible reverse proxy gateway")
    serve_parser.add_argument("--port", type=int, default=8080, help="Local listening port")
    serve_parser.add_argument("--upstream", type=str, default="https://api.openai.com/v1", help="Upstream LLM base URL")
    serve_parser.add_argument("--threshold", type=float, default=0.35, help="R_sc uncertainty threshold")
    serve_parser.add_argument("--block", action="store_true", default=False, help="Reject unsafe outputs with HTTP 422")
    serve_parser.add_argument("--k", type=int, default=3, help="Default number of paths to sample")

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Directly evaluate uncertainty on candidate answers")
    eval_parser.add_argument("samples", nargs="+", help="Candidate answer strings")
    eval_parser.add_argument("--context", type=str, default=None, help="Optional context prompt")
    eval_parser.add_argument("--threshold", type=float, default=0.35, help="R_sc uncertainty threshold")

    # Version
    subparsers.add_parser("version", help="Show Spanda version and architecture details")

    args = parser.parse_args()

    if args.command == "serve":
        run_gateway(
            port=args.port,
            upstream=args.upstream,
            threshold=args.threshold,
            block_mode=args.block,
            default_k=args.k,
        )
    elif args.command == "eval":
        guard = CascadedGuardrail(uncertainty_threshold=args.threshold)
        receipt = guard.evaluate(sampled_responses=args.samples, context=args.context)
        import json
        print(json.dumps(receipt.to_dict(), indent=2))
    elif args.command == "version":
        print("Spanda Suite v0.3.0 (Spanda Research)")
        print("Epistemic Uncertainty Engine & High-Performance LLM Gateway")
        print("DOI: 10.5281/zenodo.22233648")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
