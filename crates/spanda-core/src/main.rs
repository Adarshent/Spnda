//! Spanda Standalone CLI & High-Performance Gateway (Spanda Research).
//!
//! Sub-microsecond epistemic uncertainty quantification & OpenAI-compatible gateway.

use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::time::Instant;

use spanda_core::gateway::{create_router, GatewayState};
use spanda_core::math::SpandaEngine;

#[derive(Parser)]
#[command(name = "spnda", author = "Bhupen Nayak <bhupennayak@icloud.com>", version = "0.3.0", about = "Spanda — High-Performance Rust Epistemic Guardrail Gateway (Spanda Research)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the high-throughput OpenAI-compatible reverse-proxy gateway
    Serve {
        /// Local port to bind the gateway to
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Upstream LLM endpoint URL (OpenAI, Ollama, vLLM, Groq, etc.)
        #[arg(short, long, default_value = "https://api.openai.com")]
        upstream: String,

        /// R_sc Epistemic uncertainty rejection threshold
        #[arg(short, long, default_value_t = 0.35)]
        threshold: f64,

        /// Reject unsafe responses with HTTP 422 Unprocessable Entity
        #[arg(short, long, default_value_t = false)]
        block: bool,

        /// Number of candidate paths K to sample if client sends n=1
        #[arg(short, long, default_value_t = 3)]
        k: usize,
    },

    /// Run immediate epistemic evaluation on candidate answer strings
    Eval {
        /// Candidate response strings to evaluate
        samples: Vec<String>,

        /// Optional context prompt to evaluate grounding
        #[arg(short, long)]
        context: Option<String>,

        /// Uncertainty rejection threshold
        #[arg(short, long, default_value_t = 0.35)]
        threshold: f64,
    },

    /// Run micro-benchmarks measuring kernel latency and throughput
    Bench {
        /// Number of evaluation iterations to run
        #[arg(short, long, default_value_t = 100_000)]
        iterations: usize,
    },

    /// Show version, architecture, and theoretical foundation
    Version,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            port,
            upstream,
            threshold,
            block,
            k,
        } => {
            let state = GatewayState::new(upstream.clone(), threshold, block, k);
            let app = create_router(state);

            let addr = SocketAddr::from(([0, 0, 0, 0], port));
            println!("╔════════════════════════════════════════════════════════════════════════════╗");
            println!("║  ⚡ SPANDA ENTERPRISE GATEWAY (Spanda Research) — Pure Rust Sub-µs Engine   ║");
            println!("╚════════════════════════════════════════════════════════════════════════════╝");
            println!("→ Listening on       : http://0.0.0.0:{}", port);
            println!("→ Upstream Endpoint  : {}", upstream);
            println!("→ R_sc Threshold     : {}", threshold);
            println!("→ Enforcement Mode   : {}", if block { "STRICT BLOCK (HTTP 422)" } else { "TELEMETRY / INJECTION" });
            println!("→ Sample Paths (K)   : {}", k);
            println!("→ Telemetry / Metric : http://0.0.0.0:{}/metrics", port);
            println!("→ Health Checks      : http://0.0.0.0:{}/healthz", port);
            println!("→ Direct Eval API    : http://0.0.0.0:{}/v1/spanda/evaluate", port);
            println!("──────────────────────────────────────────────────────────────────────────────");

            let listener = tokio::net::TcpListener::bind(addr).await.unwrap_or_else(|e| {
                eprintln!("Failed to bind to address {}: {}", addr, e);
                std::process::exit(1);
            });

            axum::serve(listener, app).await.unwrap();
        }

        Commands::Eval {
            samples,
            context,
            threshold,
        } => {
            if samples.is_empty() {
                eprintln!("Error: Please provide at least 1 sample string to evaluate.");
                std::process::exit(1);
            }

            let receipt = SpandaEngine::evaluate(&samples, context.as_deref(), threshold);
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
        }

        Commands::Bench { iterations } => {
            println!("Running Spanda mathematical kernel benchmark ({} iterations)...", iterations);
            let samples = vec![
                "42".to_string(),
                "42.0".to_string(),
                "The answer is 42".to_string(),
                "43".to_string(),
                "42".to_string(),
            ];

            // Warmup
            for _ in 0..1000 {
                let _ = SpandaEngine::evaluate(&samples, None, 0.35);
            }

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = SpandaEngine::evaluate(&samples, None, 0.35);
            }
            let total_time = start.elapsed();
            let total_nanos = total_time.as_nanos();
            let per_op_nanos = total_nanos as f64 / iterations as f64;
            let ops_per_sec = (iterations as f64 / total_time.as_secs_f64()) as u64;

            println!("✓ Total Time        : {:.3?}", total_time);
            println!("✓ Latency per Eval  : {:.1} nanoseconds ({:.3} microseconds)", per_op_nanos, per_op_nanos / 1000.0);
            println!("✓ Throughput        : {} evaluations/sec on single core", ops_per_sec);
        }

        Commands::Version => {
            println!("Spanda Enterprise Suite v0.3.0 (Spanda Research)");
            println!("Architecture: Sub-Microsecond Epistemic State & Uncertainty Kernel");
            println!("- Discrete Equivalence Class Partitioning (Exact-Match Normalized Entropy)");
            println!("- 7-State Epistemic Decision Engine (Divergence & Consensus Resolution)");
            println!("- Non-Linear Attractor Basin Curvature & Mode Collapse Defense");
            println!("- Multi-Path Syllogistic Reasoning Coherence Verification");
            println!("Author: Bhupen Nayak (bhupennayak@icloud.com)");
        }
    }
}
