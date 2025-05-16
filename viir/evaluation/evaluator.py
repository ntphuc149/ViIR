"""
Evaluation module for IR models.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, evaluation
from sklearn.metrics import ndcg_score, precision_score, recall_score, accuracy_score

logger = logging.getLogger(__name__)


class IRModelEvaluator:
    """Evaluator for IR models."""
    
    def __init__(self, model: SentenceTransformer, data: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            model: SentenceTransformer model to evaluate
            data: Dataset dictionary
            config: Configuration dictionary
        """
        self.model = model
        self.data = data
        self.config = config
        self.output_dir = config["evaluation"]["output_dir"]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create corpus dict
        self.corpus_dict = {doc["id"]: doc["text"] for doc in data["corpus"]}
    
    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Evaluate the model on the specified split.
        
        Args:
            split: Data split to evaluate on ("train", "dev", or "test")
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {split} split")
        
        # Get query IDs for this split
        query_ids = self.data["splits"][split]
        
        # Filter queries
        queries = {q["id"]: q["text"] for q in self.data["queries"] if q["id"] in query_ids}
        
        # Create relevant docs mapping
        relevant_docs = {}
        for qrel in self.data["qrels"]:
            if qrel["query_id"] in query_ids:
                if qrel["query_id"] not in relevant_docs:
                    relevant_docs[qrel["query_id"]] = []
                relevant_docs[qrel["query_id"]].append(qrel["doc_id"])
        
        # Create evaluator
        evaluator = evaluation.InformationRetrievalEvaluator(
            queries=queries,
            corpus=self.corpus_dict,
            relevant_docs=relevant_docs,
            name=f"{split}-evaluation",
            batch_size=self.config["evaluation"]["batch_size"],
            show_progress_bar=True
        )
        
        # Run evaluation
        results = evaluator(self.model)
        
        # Add additional metrics
        # Extension point for custom metrics
        
        # Save results
        self._save_results(results, split)
        
        # Log results
        for metric, value in results.items():
            logger.info(f"{split} {metric}: {value:.4f}")
        
        return results
    
    def _save_results(self, results: Dict[str, float], split: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Dictionary of evaluation metrics
            split: Data split name
        """
        output_path = os.path.join(self.output_dir, f"{split}_results.json")
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved evaluation results to {output_path}")
    
    def evaluate_all_splits(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on all splits.
        
        Returns:
            Dictionary mapping split names to metric dictionaries
        """
        all_results = {}
        
        for split in ["train", "dev", "test"]:
            all_results[split] = self.evaluate(split)
        
        return all_results


def evaluate_model(model_path: str, data_dir: str, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a saved model.
    
    Args:
        model_path: Path to the saved model
        data_dir: Directory containing the processed data
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation results for all splits
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load data
    from data.processor import load_processed_data
    data = load_processed_data(data_dir)
    
    # Create evaluator
    evaluator = IRModelEvaluator(model, data, config)
    
    # Evaluate on all splits
    results = evaluator.evaluate_all_splits()
    
    return results