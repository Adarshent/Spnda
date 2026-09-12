//! 7-State Epistemic Consensus State Machine.
//!
//! Maps generative model uncertainty into 7 deterministic consensus states,
//! treating truth not as a naive binary, but as a conditioned manifold
//! of evidential consistency across multi-path generations.

use serde::{Serialize, Deserialize};

/// The seven deterministic epistemic consensus states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicState {
    /// 1. High unanimous or overwhelming consensus (R_sc <= 0.15).
    #[serde(rename = "CONSISTENT")]
    Consistent,

    /// 2. High epistemic divergence / severe dispersion (R_sc > 0.45).
    #[serde(rename = "HIGH_UNCERTAINTY")]
    HighUncertainty,

    /// 3. Bimodal split between two strong contradictory hypotheses.
    #[serde(rename = "BIMODAL_SPLIT")]
    BimodalSplit,

    /// 4. Degenerate Attractor Basin or Confident Mode Collapse detected.
    #[serde(rename = "MODE_COLLAPSE")]
    ModeCollapse,

    /// 5. Weak majority consensus coupled with attractor basin resonance.
    #[serde(rename = "DOMINANT_WITH_ATTRACTOR")]
    DominantWithAttractor,

    /// 6. Multimodal dispersion with an unnatural attractor signature.
    #[serde(rename = "DISPERSED_WITH_ATTRACTOR")]
    DispersedWithAttractor,

    /// 7. Complete chaotic breakdown across paths.
    #[serde(rename = "CHAOTIC_DISPERSION")]
    ChaoticDispersion,
}

impl EpistemicState {
    /// Returns the standard status string for external APIs and headers.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Consistent => "CONSISTENT",
            Self::HighUncertainty => "HIGH_UNCERTAINTY",
            Self::BimodalSplit => "BIMODAL_SPLIT",
            Self::ModeCollapse => "MODE_COLLAPSE",
            Self::DominantWithAttractor => "DOMINANT_WITH_ATTRACTOR",
            Self::DispersedWithAttractor => "DISPERSED_WITH_ATTRACTOR",
            Self::ChaoticDispersion => "CHAOTIC_DISPERSION",
        }
    }

    /// Translates the epistemic state into an enterprise guardrail decision.
    pub fn to_decision(&self) -> (&'static str, bool) {
        match self {
            Self::Consistent => ("FAST_PASS_CONSISTENT", true),
            Self::DominantWithAttractor => ("PASS_WITH_WARNING", true),
            Self::BimodalSplit => ("REJECT_BIMODAL_UNCERTAINTY", false),
            Self::ModeCollapse => ("REJECT_ATTRACTOR_COLLAPSE", false),
            Self::HighUncertainty => ("REJECT_HIGH_UNCERTAINTY", false),
            Self::DispersedWithAttractor => ("REJECT_DISPERSED_COLLAPSE", false),
            Self::ChaoticDispersion => ("REJECT_CHAOTIC_DISPERSION", false),
        }
    }
}

/// Evaluates the epistemic state of K responses given R_sc and attractor signals.
pub fn classify_epistemic_state(
    rsc: f64,
    agreement_ratio: f64,
    distinct_partitions: usize,
    attractor_signature: bool,
    threshold: f64,
) -> EpistemicState {
    // Check for Confident Mode Collapse / Attractor Basin trap
    if attractor_signature {
        if agreement_ratio >= 0.8 {
            return EpistemicState::ModeCollapse;
        } else if agreement_ratio >= 0.5 {
            return EpistemicState::DominantWithAttractor;
        } else {
            return EpistemicState::DispersedWithAttractor;
        }
    }

    // High consensus: R_sc <= threshold and strong agreement
    if rsc <= threshold && agreement_ratio >= 0.6 {
        return EpistemicState::Consistent;
    }

    // Bimodal split: exactly 2 dominant partitions of roughly equal weight (e.g. 50/50 or 60/40)
    if distinct_partitions == 2 && (0.4..=0.6).contains(&agreement_ratio) {
        return EpistemicState::BimodalSplit;
    }

    // High entropy / dispersion
    if rsc > 0.60 {
        if distinct_partitions >= 3 && agreement_ratio < 0.35 {
            return EpistemicState::ChaoticDispersion;
        }
        return EpistemicState::HighUncertainty;
    }

    // Borderline cases
    if rsc > threshold {
        EpistemicState::HighUncertainty
    } else {
        EpistemicState::Consistent
    }
}
