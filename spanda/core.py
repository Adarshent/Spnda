import re
import math
import collections
from typing import List, Dict, Any, Union

def normalize_answer(text: str) -> str:
    """
    Deterministically normalizes a string for exact-match equivalence.
    Lowercases, strips punctuation, and normalizes whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def compute_rsc(answers: List[str], alpha: float = 0.5) -> Dict[str, Any]:
    """
    Computes the Spanda (R_sc) exact-match entropy risk score for a set of sampled answers.
    
    Args:
        answers: A list of K string answers sampled from an LLM.
        alpha: The weighting factor for H_norm vs (1 - w_max). Default 0.5.
        
    Returns:
        A dictionary containing the rsc score, components, and the dominant answer.
    """
    if not answers:
        raise ValueError("The answers list cannot be empty.")
        
    K = len(answers)
    
    # 1. Normalize and cluster
    normalized_answers = [normalize_answer(a) for a in answers]
    counts = collections.Counter(normalized_answers)
    
    # 2. Compute probabilities (w_i)
    w = [count / K for count in counts.values()]
    n_clusters = len(counts)
    
    # 3. Compute normalized entropy H_norm
    if n_clusters == 1:
        h_norm = 0.0
    else:
        # Avoid math domain error by capping denominator
        denom = math.log(K) if K > 1 else 1.0
        h_norm = sum(-p * math.log(p) for p in w) / denom
        
    # 4. Compute modal dominance (w_max)
    w_max = max(w)
    
    # 5. Compute R_sc
    rsc = alpha * h_norm + (1 - alpha) * (1 - w_max)
    
    # Find the string representation of the dominant cluster
    dominant_cluster_norm = counts.most_common(1)[0][0]
    
    # Find original string that mapped to dominant cluster (first occurrence)
    dominant_answer = next(
        ans for ans, norm in zip(answers, normalized_answers) 
        if norm == dominant_cluster_norm
    )
    
    return {
        "rsc": round(rsc, 4),
        "h_norm": round(h_norm, 4),
        "w_max": round(w_max, 4),
        "n_clusters": n_clusters,
        "dominant_answer": dominant_answer,
    }

def detect_hallucination(answers: List[str], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Convenience function that computes R_sc and returns a boolean flagging 
    whether the model is likely hallucinating (R_sc > threshold).
    
    NOTE: On models > 100B params doing factual recall, confident mode collapse
    may cause R_sc to be near 0 even for hallucinations. Use with caution on frontier models.
    """
    result = compute_rsc(answers)
    result["is_uncertain"] = result["rsc"] > threshold
    return result

def batch_compute_rsc(batch_answers: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Computes R_sc for a batch of sampled generations.
    """
    return [compute_rsc(answers) for answers in batch_answers]
