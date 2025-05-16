"""
Custom metrics for IR evaluation.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def compute_mrr(rankings: List[List[int]]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        rankings: List of rankings, where each ranking is a list of 0/1 indicating relevance
                 at each position (1 = relevant, 0 = not relevant)
    
    Returns:
        Mean Reciprocal Rank score
    """
    reciprocal_ranks = []
    
    for ranking in rankings:
        # Find the first relevant document
        try:
            first_rel_pos = ranking.index(1) + 1  # +1 for 1-based indexing
            reciprocal_ranks.append(1.0 / first_rel_pos)
        except ValueError:  # No relevant document found
            reciprocal_ranks.append(0.0)
    
    if not reciprocal_ranks:
        return 0.0
    
    return np.mean(reciprocal_ranks)


def compute_precision_at_k(rankings: List[List[int]], k: int = 10) -> float:
    """
    Compute Precision@k.
    
    Args:
        rankings: List of rankings, where each ranking is a list of 0/1 indicating relevance
        k: Cutoff position
    
    Returns:
        Precision@k score
    """
    precision_scores = []
    
    for ranking in rankings:
        # Consider only top-k documents
        top_k = ranking[:k]
        # Pad with zeros if fewer than k documents
        top_k = top_k + [0] * (k - len(top_k))
        
        # Compute precision
        if sum(top_k) > 0:
            precision_scores.append(sum(top_k) / k)
        else:
            precision_scores.append(0.0)
    
    if not precision_scores:
        return 0.0
    
    return np.mean(precision_scores)


def compute_recall_at_k(rankings: List[List[int]], relevant_counts: List[int], k: int = 10) -> float:
    """
    Compute Recall@k.
    
    Args:
        rankings: List of rankings, where each ranking is a list of 0/1 indicating relevance
        relevant_counts: List of total relevant documents for each query
        k: Cutoff position
    
    Returns:
        Recall@k score
    """
    recall_scores = []
    
    for ranking, rel_count in zip(rankings, relevant_counts):
        if rel_count == 0:
            continue
            
        # Consider only top-k documents
        top_k = ranking[:k]
        
        # Compute recall
        recall_scores.append(sum(top_k) / rel_count)
    
    if not recall_scores:
        return 0.0
    
    return np.mean(recall_scores)


def compute_f1_at_k(rankings: List[List[int]], relevant_counts: List[int], k: int = 10) -> float:
    """
    Compute F1@k.
    
    Args:
        rankings: List of rankings, where each ranking is a list of 0/1 indicating relevance
        relevant_counts: List of total relevant documents for each query
        k: Cutoff position
    
    Returns:
        F1@k score
    """
    precision = compute_precision_at_k(rankings, k)
    recall = compute_recall_at_k(rankings, relevant_counts, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_metrics(rankings: List[List[int]], 
                   relevant_counts: List[int],
                   cutoffs: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Compute various IR metrics at different cutoffs.
    
    Args:
        rankings: List of rankings, where each ranking is a list of 0/1 indicating relevance
        relevant_counts: List of total relevant documents for each query
        cutoffs: List of cutoff positions
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MRR is independent of cutoff
    metrics["mrr"] = compute_mrr(rankings)
    
    # Compute metrics at each cutoff
    for k in cutoffs:
        metrics[f"precision@{k}"] = compute_precision_at_k(rankings, k)
        metrics[f"recall@{k}"] = compute_recall_at_k(rankings, relevant_counts, k)
        metrics[f"f1@{k}"] = compute_f1_at_k(rankings, relevant_counts, k)
    
    return metrics