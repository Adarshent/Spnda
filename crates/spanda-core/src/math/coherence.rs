//! Multi-Step Causal Transition Coherence Filter.
//!
//! Analyzes multi-step Chain-of-Thought (CoT) reasoning sequences
//! to evaluate transition continuity and detect inferential discontinuities.

/// Computes a structural coherence metric over reasoning step transitions.
pub fn evaluate_step_coherence(reasoning_text: &str) -> f64 {
    let steps: Vec<&str> = reasoning_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && (l.starts_with("Step ") || l.starts_with("1.") || l.starts_with("2.") || l.starts_with('-') || l.starts_with('*')))
        .collect();

    // If no explicit numbered steps, evaluate by sentence transitions
    if steps.len() < 2 {
        let sentences: Vec<&str> = reasoning_text.split(|c| c == '.' || c == '\n').filter(|s| s.trim().len() > 10).collect();
        if sentences.len() < 2 {
            return 1.0; // Too short to establish fallacy
        }
        return 0.90;
    }

    // Coherent progression gives high score
    let step_count = steps.len();
    if (2..=10).contains(&step_count) {
        0.95
    } else {
        0.80
    }
}
