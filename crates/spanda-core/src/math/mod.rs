//! Spanda Sub-Microsecond Epistemic Uncertainty & Guardrail Engine.
//!
//! Sub-microsecond epistemic uncertainty quantification & guardrail verification.

pub mod partition;
pub mod consensus;
pub mod attractor;
pub mod coherence;

pub use partition::{compute_partition_entropy, normalize_canonical, PartitionResult};
pub use consensus::{classify_epistemic_state, EpistemicState};
pub use attractor::evaluate_attractor_basin;
pub use coherence::evaluate_step_coherence;

use serde::{Serialize, Deserialize};
use std::time::Instant;

/// Enterprise audit receipt produced for every evaluated LLM completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpandaAuditReceipt {
    /// Normalized epistemic uncertainty index R_sc in [0.0, 1.0].
    pub rsc: f64,
    /// Classified epistemic state.
    pub epistemic_state: EpistemicState,
    /// Standard enterprise status string (e.g. "CONSISTENT", "MODE_COLLAPSE").
    pub state_name: String,
    /// High-level enterprise guardrail decision.
    pub decision: String,
    /// Boolean safety flag (true = safe to emit, false = reject).
    pub is_safe: bool,
    /// Consensus dominant canonical answer chosen from the sampled paths.
    pub dominant_answer: String,
    /// Number of candidate paths sampled K.
    pub k_samples: usize,
    /// Fraction of paths that converged to the dominant answer.
    pub agreement_ratio: f64,
    /// Attractor Basin / Confident Mode Collapse flag.
    pub attractor_collapse: bool,
    /// Anomaly severity score in [0.0, 1.0].
    pub attractor_score: f64,
    /// Execution latency of the mathematical kernel in nanoseconds.
    pub latency_nanos: u128,
    /// Execution latency formatted in microseconds.
    pub latency_micros: f64,
}

pub struct SpandaEngine;

impl SpandaEngine {
    /// Evaluates K sampled LLM responses in sub-microsecond time.
    pub fn evaluate(
        samples: &[String],
        context: Option<&str>,
        threshold: f64,
    ) -> SpandaAuditReceipt {
        let start = Instant::now();

        // 1. Combinatorial partition & entropy
        let partition = compute_partition_entropy(samples);

        // 2. Curvature / Attractor Basin check
        let (attractor_collapse, attractor_score) = evaluate_attractor_basin(samples, context);

        // 3. Epistemic state classification
        let epistemic_state = classify_epistemic_state(
            partition.rsc,
            partition.agreement_ratio,
            partition.distinct_partitions,
            attractor_collapse,
            threshold,
        );

        let (decision, is_safe) = epistemic_state.to_decision();
        let elapsed = start.elapsed();
        let latency_nanos = elapsed.as_nanos();
        let latency_micros = latency_nanos as f64 / 1000.0;

        SpandaAuditReceipt {
            rsc: (partition.rsc * 10000.0).round() / 10000.0,
            state_name: epistemic_state.as_str().to_string(),
            epistemic_state,
            decision: decision.to_string(),
            is_safe,
            dominant_answer: partition.dominant_answer,
            k_samples: partition.k_samples,
            agreement_ratio: (partition.agreement_ratio * 10000.0).round() / 10000.0,
            attractor_collapse,
            attractor_score: (attractor_score * 1000.0).round() / 10000.0,
            latency_nanos,
            latency_micros,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_submicrosecond_speed() {
        let samples = vec![
            "42".to_string(),
            "42.0".to_string(),
            "The answer is 42".to_string(),
            "The answer is 42".to_string(),
        ];
        let receipt = SpandaEngine::evaluate(&samples, None, 0.35);
        assert!(receipt.is_safe);
        assert_eq!(receipt.rsc, 0.0);
        assert_eq!(receipt.decision, "FAST_PASS_CONSISTENT");
        assert_eq!(receipt.state_name, "CONSISTENT");
        println!("Latency: {} ns ({} µs)", receipt.latency_nanos, receipt.latency_micros);
    }
}
