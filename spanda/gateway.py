"""
Spanda Gateway (BRHMN Guardrail Proxy).

A zero-dependency, ultra-low latency OpenAI-compatible reverse proxy that injects
sub-millisecond epistemic uncertainty quantification ($R_{sc}$) and 2-Tier cascaded
guardrail verification into every chat completion request.

Works out of the box with OpenAI, Groq, vLLM, Ollama, Together, and Mistral.

Usage:
    python -m spanda.gateway --upstream https://api.openai.com/v1 --port 8080

Client setup:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-...")
"""

import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, List, Optional

from spanda.core import compute_rsc
from spanda.guardrails import CascadedGuardrail, AuditReceipt


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SpandaGatewayHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler that proxies OpenAI-compatible requests and executes
    Spanda epistemic uncertainty checks on responses.
    """
    upstream_url: str = "https://api.openai.com/v1"
    guardrail: CascadedGuardrail = CascadedGuardrail(uncertainty_threshold=0.35)
    block_mode: bool = False
    default_k: int = 3

    def do_GET(self):
        """Handle health checks and pass-through GET requests."""
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, {
                "status": "healthy",
                "service": "spanda-gateway",
                "engine": "Spanda v0.2.0 (BRHMN Labs)",
                "upstream": self.upstream_url,
                "block_mode": self.block_mode,
            })
            return

        # Pass-through for /v1/models etc.
        self._proxy_request("GET")

    def do_POST(self):
        """Handle completions and chat completions."""
        if self.path.endswith("/chat/completions") or self.path.endswith("/completions"):
            self._handle_completions()
        else:
            self._proxy_request("POST")

    def _handle_completions(self):
        """Intercepts chat completions, evaluates R_sc, and injects guardrail telemetry."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        try:
            req_data = json.loads(post_body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "Invalid JSON payload"})
            return

        # Extract context if present in messages or tools
        context = self._extract_context(req_data)

        # If client requested n=1, we can adjust or preserve n
        # If sampling is enabled and n is not set, set n = default_k to get epistemic samples
        original_n = req_data.get("n", 1)
        if original_n == 1 and self.default_k > 1 and not req_data.get("stream", False):
            req_data["n"] = self.default_k

        # Forward modified request to upstream
        forward_body = json.dumps(req_data).encode("utf-8")
        headers = self._filter_headers(dict(self.headers))
        headers["Content-Length"] = str(len(forward_body))

        target_url = self.upstream_url.rstrip("/") + self.path
        if not target_url.endswith("/chat/completions") and "/chat/completions" in self.path:
            target_url = self.upstream_url.rstrip("/") + "/chat/completions"

        req = urllib.request.Request(target_url, data=forward_body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                resp_status = resp.status
                resp_bytes = resp.read()
                resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            err_bytes = e.read()
            self._send_raw(e.code, err_bytes, dict(e.headers))
            return
        except Exception as e:
            self._send_json(502, {"error": f"Upstream proxy failed: {str(e)}"})
            return

        # Parse upstream response
        try:
            resp_data = json.loads(resp_bytes.decode("utf-8"))
        except Exception:
            # If not JSON (or streaming), return as-is
            self._send_raw(resp_status, resp_bytes, resp_headers)
            return

        # Evaluate choices with Spanda
        choices = resp_data.get("choices", [])
        if len(choices) >= 2:
            samples = []
            for c in choices:
                msg = c.get("message", {})
                content = msg.get("content") or c.get("text", "")
                if content:
                    samples.append(content)

            if samples:
                start_t = time.perf_counter()
                receipt: AuditReceipt = self.guardrail.evaluate(sampled_responses=samples, context=context)
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                extra_headers = {
                    "X-Spanda-Rsc": str(round(receipt.rsc, 4)),
                    "X-Spanda-Safe": "true" if receipt.is_safe else "false",
                    "X-Spanda-Decision": receipt.decision,
                    "X-Spanda-Confidence": str(round(receipt.confidence, 4)),
                    "X-Spanda-Latency-Ms": f"{latency_ms:.3f}",
                    "X-Spanda-Tier": str(receipt.tier_executed),
                }

                # If original request only wanted 1 choice, prune choices down to the dominant answer
                if original_n == 1:
                    dominant = receipt.dominant_answer
                    # Find matching choice or keep choice 0 with dominant answer
                    best_choice = choices[0]
                    for c in choices:
                        c_text = (c.get("message", {}).get("content") or c.get("text", ""))
                        if c_text.strip() == dominant.strip():
                            best_choice = c
                            break
                    resp_data["choices"] = [best_choice]

                # Inject Spanda audit metadata directly into response object
                resp_data["spanda_audit"] = receipt.to_dict()

                # If block_mode is True and receipt is unsafe, return 422
                if self.block_mode and not receipt.is_safe:
                    self._send_json(422, {
                        "error": "Spanda Guardrail Rejection: Hallucination or Mode Collapse Detected",
                        "audit": receipt.to_dict()
                    }, extra_headers=extra_headers)
                    return

                new_body = json.dumps(resp_data).encode("utf-8")
                self._send_raw(resp_status, new_body, extra_headers=extra_headers)
                return

        # Default fallback: return response as-is
        self._send_raw(resp_status, resp_bytes, resp_headers)

    def _extract_context(self, req_data: Dict[str, Any]) -> Optional[str]:
        """Extracts context from system prompt or developer messages."""
        messages = req_data.get("messages", [])
        contexts = []
        for m in messages:
            if m.get("role") in ("system", "developer"):
                contexts.append(m.get("content", ""))
        return " ".join(contexts) if contexts else None

    def _proxy_request(self, method: str):
        """Standard reverse proxy for non-intercepted endpoints."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        target_url = self.upstream_url.rstrip("/") + self.path
        headers = self._filter_headers(dict(self.headers))
        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req) as resp:
                self._send_raw(resp.status, resp.read(), dict(resp.headers))
        except urllib.error.HTTPError as e:
            self._send_raw(e.code, e.read(), dict(e.headers))
        except Exception as e:
            self._send_json(502, {"error": str(e)})

    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Strip hop-by-hop headers before forwarding."""
        filtered = {}
        for k, v in headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                filtered[k] = v
        return filtered

    def _send_json(self, status: int, data: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, status: int, body: bytes, extra_headers: Optional[Dict[str, str]] = None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quiet default logging
        pass


def run_gateway(
    port: int = 8080,
    upstream: str = "https://api.openai.com/v1",
    threshold: float = 0.35,
    block_mode: bool = False,
    default_k: int = 3,
):
    """Starts the Spanda Gateway server."""
    SpandaGatewayHandler.upstream_url = upstream
    SpandaGatewayHandler.guardrail = CascadedGuardrail(uncertainty_threshold=threshold)
    SpandaGatewayHandler.block_mode = block_mode
    SpandaGatewayHandler.default_k = default_k

    server = ThreadingHTTPServer(("0.0.0.0", port), SpandaGatewayHandler)
    print(f"⚡ Spanda Gateway (BRHMN Labs) active on http://0.0.0.0:{port}")
    print(f"→ Upstream LLM: {upstream}")
    print(f"→ Guardrail Threshold (R_sc): {threshold}")
    print(f"→ Block Mode: {'ENABLED (422 rejection)' if block_mode else 'DISABLED (Header telemetry)'}")
    print(f"→ Sampling paths (K): {default_k}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Spanda Gateway.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Spanda Guardrail Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Local listening port (default: 8080)")
    parser.add_argument("--upstream", type=str, default="https://api.openai.com/v1", help="Upstream API base URL")
    parser.add_argument("--threshold", type=float, default=0.35, help="R_sc uncertainty threshold (default: 0.35)")
    parser.add_argument("--block", action="store_true", help="Reject unsafe responses with HTTP 422")
    parser.add_argument("--k", type=int, default=3, help="Default number of paths to sample (default: 3)")
    args = parser.parse_args()

    run_gateway(
        port=args.port,
        upstream=args.upstream,
        threshold=args.threshold,
        block_mode=args.block,
        default_k=args.k,
    )


if __name__ == "__main__":
    main()
