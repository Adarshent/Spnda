//! Dynamic Attractor Basin & Mode Collapse Detector.
//!
//! Evaluates trajectory curvature and degenerate stability to counter
//! Confident Mode Collapse on high-capacity frontier models, where
//! networks collapse unanimously into a single incorrect attractor basin.

use std::collections::HashSet;

/// Evaluates whether a set of unanimous or near-unanimous samples is trapped
/// in a degenerate Attractor Basin or Confident Mode Collapse loop.
pub fn evaluate_attractor_basin(
    samples: &[String],
    context: Option<&str>,
) -> (bool, f64) {
    if samples.is_empty() {
        return (false, 0.0);
    }

    let mut anomaly_score: f64 = 0.0;

    // Check 1: Degenerate repetitive loop detection (token trigram entropy)
    let sample = &samples[0];
    let words: Vec<&str> = sample.split_whitespace().collect();
    if words.len() >= 12 {
        let mut trigrams: HashSet<(&str, &str, &str)> = HashSet::new();
        let total_trigrams = words.len().saturating_sub(2);
        for i in 0..total_trigrams {
            trigrams.insert((words[i], words[i + 1], words[i + 2]));
        }
        let trigram_ratio = trigrams.len() as f64 / total_trigrams.max(1) as f64;
        // Unusually low trigram diversity indicates a cyclic degenerative loop
        if trigram_ratio < 0.40 {
            anomaly_score += 0.55;
        }
    }

    // Check 2: Canned evasive boilerplate collapse
    let lower = sample.to_lowercase();
    let evasive_signatures = [
        "i cannot fulfill",
        "as an ai language model",
        "i do not possess the ability",
        "as a large language model",
        "i'm sorry, but i cannot",
    ];
    for sig in evasive_signatures {
        if lower.contains(sig) {
            anomaly_score += 0.40;
            break;
        }
    }

    // Check 3: Grounding disconnect with system/retrieved context (if provided)
    if let Some(ctx) = context {
        let ctx_words: HashSet<String> = ctx
            .split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| w.len() > 3)
            .collect();

        if !ctx_words.is_empty() {
            let sample_words: HashSet<String> = words
                .iter()
                .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() > 3)
                .collect();

            let overlap = ctx_words.intersection(&sample_words).count();
            let overlap_ratio = overlap as f64 / ctx_words.len().min(10) as f64;

            // If context provided facts, but generation completely diverged into vacuum
            if overlap_ratio < 0.05 && words.len() > 20 {
                anomaly_score += 0.35;
            }
        }
    }

    let is_collapse = anomaly_score >= 0.50;
    (is_collapse, anomaly_score.min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_response() {
        let samples = vec![
            "The capital of France is Paris.".to_string(),
            "Paris is the capital of France.".to_string(),
        ];
        let (collapse, score) = evaluate_attractor_basin(&samples, Some("Geography trivia about France"));
        assert!(!collapse);
        assert!(score < 0.5);
    }

    #[test]
    fn test_repetitive_loop_collapse() {
        let loop_text = "error in system error in system error in system error in system error in system error in system error in system";
        let samples = vec![loop_text.to_string()];
        let (collapse, score) = evaluate_attractor_basin(&samples, None);
        assert!(collapse);
        assert!(score >= 0.5);
    }
}
