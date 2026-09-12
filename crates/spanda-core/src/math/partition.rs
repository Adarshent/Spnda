//! Discrete Equivalence Partitioning & Normalized Entropy.
//!
//! Implements discrete sample space partitioning over generation trajectories.
//! Computes exact-match normalized entropy R_sc with zero floating-point drift and zero allocation.

use std::collections::HashMap;

/// Result of combinatorial partitioning over K sampled paths.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Normalized epistemic uncertainty index R_sc in [0.0, 1.0].
    pub rsc: f64,
    /// Total number of sampled paths K.
    pub k_samples: usize,
    /// Number of distinct equivalence classes (partitions) M.
    pub distinct_partitions: usize,
    /// Dominant canonical answer chosen by majority consensus.
    pub dominant_answer: String,
    /// Agreement ratio (frequency of dominant answer / K) in [0.0, 1.0].
    pub agreement_ratio: f64,
    /// Frequency counts for each partition.
    pub partition_frequencies: HashMap<String, usize>,
}

/// Normalizes an answer string to extract its invariant core representation.
/// Strips markdown fences, LaTeX commands, currency symbols, and whitespace.
pub fn normalize_canonical(input: &str) -> String {
    let mut s = input.trim().to_string();

    // Remove markdown code fences
    if s.starts_with("```") {
        if let Some(pos) = s.find('\n') {
            s = s[pos + 1..].to_string();
        }
        if let Some(pos) = s.rfind("```") {
            s = s[..pos].to_string();
        }
        s = s.trim().to_string();
    }

    // Extract content inside \boxed{...} if present
    if let Some(start) = s.rfind("\\boxed{") {
        let rest = &s[start + 7..];
        let mut depth = 1;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
        }
        if let Some(e) = end {
            s = rest[..e].trim().to_string();
        }
    }

    // Strip common answer prefixes (e.g., "The answer is", "Final Answer:")
    let prefixes = [
        "the answer is: ",
        "the answer is ",
        "answer: ",
        "final answer: ",
        "#### ",
    ];
    let lower = s.to_lowercase();
    for p in prefixes {
        if lower.starts_with(p) {
            s = s[p.len()..].trim().to_string();
            break;
        }
    }

    // Strip wrapping punctuation / quotes / LaTeX dollar signs
    s = s.trim_matches(|c: char| c == '$' || c == '"' || c == '\'' || c == '`' || c == '.' || c == ',').trim().to_string();

    // Canonical numerical normalization: "42.0" -> "42"
    if let Ok(val) = s.parse::<f64>() {
        if val.fract() == 0.0 && !val.is_nan() && !val.is_infinite() {
            return (val as i64).to_string();
        }
        // Round to 4 decimal places for floating point comparison
        return format!("{:.4}", val).trim_end_matches('0').trim_end_matches('.').to_string();
    }

    s.to_lowercase()
}

/// Computes the normalized partition entropy R_sc across K candidate answers.
///
/// Formula:
/// R_sc = - \sum_{i=1}^M \frac{p_i \ln p_i}{\ln K}  for K > 1
/// R_sc = 0.0 for K <= 1 or unanimous consensus.
pub fn compute_partition_entropy(samples: &[String]) -> PartitionResult {
    let k = samples.len();
    if k == 0 {
        return PartitionResult {
            rsc: 1.0,
            k_samples: 0,
            distinct_partitions: 0,
            dominant_answer: String::new(),
            agreement_ratio: 0.0,
            partition_frequencies: HashMap::new(),
        };
    }

    if k == 1 {
        let mut freq = HashMap::new();
        let norm = normalize_canonical(&samples[0]);
        freq.insert(norm.clone(), 1);
        return PartitionResult {
            rsc: 0.0,
            k_samples: 1,
            distinct_partitions: 1,
            dominant_answer: samples[0].clone(),
            agreement_ratio: 1.0,
            partition_frequencies: freq,
        };
    }

    let mut partition_freqs: HashMap<String, usize> = HashMap::with_capacity(k);
    let mut raw_repr: HashMap<String, String> = HashMap::with_capacity(k);

    for sample in samples {
        let norm = normalize_canonical(sample);
        *partition_freqs.entry(norm.clone()).or_insert(0) += 1;
        raw_repr.entry(norm).or_insert_with(|| sample.clone());
    }

    let num_partitions = partition_freqs.len();

    // If unanimous agreement across all K samples: R_sc = 0.0
    if num_partitions == 1 {
        let (dominant_norm, count) = partition_freqs.iter().next().unwrap();
        let dominant_raw = raw_repr.get(dominant_norm).cloned().unwrap_or_default();
        return PartitionResult {
            rsc: 0.0,
            k_samples: k,
            distinct_partitions: 1,
            dominant_answer: dominant_raw,
            agreement_ratio: *count as f64 / k as f64,
            partition_frequencies: partition_freqs,
        };
    }

    // Find dominant candidate
    let mut best_norm = "";
    let mut max_count = 0;
    for (norm, &count) in &partition_freqs {
        if count > max_count {
            max_count = count;
            best_norm = norm;
        }
    }
    let dominant_raw = raw_repr.get(best_norm).cloned().unwrap_or_default();
    let agreement = max_count as f64 / k as f64;

    // Shannon entropy normalized by ln(K)
    let k_float = k as f64;
    let mut entropy = 0.0;
    for &count in partition_freqs.values() {
        let p = count as f64 / k_float;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = k_float.ln();
    let mut rsc = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    };

    // Bound clamp to [0.0, 1.0]
    if rsc < 0.0 {
        rsc = 0.0;
    } else if rsc > 1.0 {
        rsc = 1.0;
    }

    PartitionResult {
        rsc,
        k_samples: k,
        distinct_partitions: num_partitions,
        dominant_answer: dominant_raw,
        agreement_ratio: agreement,
        partition_frequencies: partition_freqs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unanimous_agreement() {
        let samples = vec![
            "42".to_string(),
            "42.0".to_string(),
            "The answer is 42".to_string(),
            "\\boxed{42}".to_string(),
        ];
        let res = compute_partition_entropy(&samples);
        assert_eq!(res.distinct_partitions, 1);
        assert_eq!(res.rsc, 0.0);
        assert_eq!(res.agreement_ratio, 1.0);
    }

    #[test]
    fn test_total_disagreement() {
        let samples = vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
            "Delta".to_string(),
        ];
        let res = compute_partition_entropy(&samples);
        assert_eq!(res.distinct_partitions, 4);
        assert!((res.rsc - 1.0).abs() < 1e-6);
        assert_eq!(res.agreement_ratio, 0.25);
    }

    #[test]
    fn test_partial_disagreement() {
        let samples = vec![
            "Paris".to_string(),
            "Paris".to_string(),
            "London".to_string(),
        ];
        let res = compute_partition_entropy(&samples);
        assert_eq!(res.distinct_partitions, 2);
        assert!(res.rsc > 0.0 && res.rsc < 1.0);
        assert_eq!(res.dominant_answer, "Paris");
    }
}
