"""
Dataset creation module for different training strategies.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IRDatasetBase:
    """Base class for IR datasets."""
    
    def __init__(self, data: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the IR dataset.
        
        Args:
            data: Processed data dictionary
            config: Configuration dictionary
        """
        self.data = data
        self.config = config
        
        # Create mapping from query ID to query text
        self.qid_to_query = {q["id"]: q["text"] for q in data["queries"]}
        
        # Create corpus mapping (doc_id -> text)
        self.corpus_dict = {doc["id"]: doc["text"] for doc in data["corpus"]}
        self.corpus_ids = [doc["id"] for doc in data["corpus"]]
        self.corpus_texts = [doc["text"] for doc in data["corpus"]]
        
        # Create mapping from query to positive docs
        self.query_to_positives = {}
        for qrel in data["qrels"]:
            if qrel["query_id"] not in self.query_to_positives:
                self.query_to_positives[qrel["query_id"]] = []
            self.query_to_positives[qrel["query_id"]].append(qrel["doc_id"])
    
    def create_train_examples(self) -> List[InputExample]:
        """
        Create training examples.
        
        Returns:
            List of InputExample objects for training
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_eval_data(self, split: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
        """
        Create evaluation data for a specific split.
        
        Args:
            split: Split name ("train", "dev", or "test")
            
        Returns:
            Tuple containing:
            - queries dict (id -> text)
            - corpus dict (id -> text)
            - relevant docs dict (query_id -> list of doc_ids)
        """
        # Get query IDs for this split
        query_ids = self.data["splits"][split]
        
        # Filter queries
        queries = {q_id: self.qid_to_query[q_id] for q_id in query_ids}
        
        # Create relevant docs mapping
        relevant_docs = {}
        for qrel in self.data["qrels"]:
            if qrel["query_id"] in query_ids:
                if qrel["query_id"] not in relevant_docs:
                    relevant_docs[qrel["query_id"]] = []
                relevant_docs[qrel["query_id"]].append(qrel["doc_id"])
        
        return queries, self.corpus_dict, relevant_docs


class PositivePairDataset(IRDatasetBase):
    """Dataset for positive pair training strategy."""
    
    def create_train_examples(self) -> List[InputExample]:
        """
        Create training examples using only positive query-document pairs.
        
        Returns:
            List of InputExample objects for training
        """
        logger.info("Creating positive pair training examples")
        
        train_examples = []
        train_queries = self.data["splits"]["train"]
        
        for query_id in tqdm(train_queries, desc="Creating positive pairs"):
            query_text = self.qid_to_query[query_id]
            positive_ids = self.query_to_positives.get(query_id, [])
            
            for doc_id in positive_ids:
                doc_text = self.corpus_dict[doc_id]
                # Thêm label=1.0 vào InputExample
                train_examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
        
        logger.info(f"Created {len(train_examples)} positive pair examples")
        return train_examples


class HardNegativeDataset(IRDatasetBase):
    """Dataset for hard negative training strategy."""
    
    def create_train_examples(self) -> List[InputExample]:
        """
        Create training examples with hard negatives.
        
        Returns:
            List of InputExample objects for training
        """
        logger.info("Creating hard negative training examples")
        
        # Parameters
        n_hard_negatives = self.config["data"].get("hard_negatives_per_query", 3)
        n_random_negatives = self.config["data"].get("random_negatives_per_query", 2)
        
        # Get initial model for hard negative mining
        model_name = self.config["hard_negative_mining"]["initial_model"]
        logger.info(f"Using {model_name} for hard negative mining")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Encode corpus
        logger.info(f"Encoding {len(self.corpus_texts)} documents")
        corpus_embeddings = model.encode(
            self.corpus_texts, 
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=self.config["hard_negative_mining"].get("mining_batch_size", 32)
        )
        
        # Create training examples with hard negatives
        train_examples = []
        train_queries = self.data["splits"]["train"]
        
        for query_id in tqdm(train_queries, desc="Mining hard negatives"):
            query_text = self.qid_to_query[query_id]
            positive_ids = self.query_to_positives.get(query_id, [])
            
            # Skip if no positive documents
            if not positive_ids:
                continue
            
            # Find positive texts
            positive_texts = [self.corpus_dict[doc_id] for doc_id in positive_ids]
            
            # Create query embedding
            query_embedding = model.encode(query_text, convert_to_tensor=True)
            
            # Calculate similarity with all documents
            cos_scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
            
            # Get hard negatives (high similarity but not positive)
            hard_negative_indices = []
            cos_scores_np = cos_scores.cpu().numpy()
            sorted_indices = np.argsort(-cos_scores_np)  # Sort in descending order
            
            # Find hard negatives
            for idx in sorted_indices:
                doc_id = self.corpus_ids[idx]
                if doc_id not in positive_ids:
                    hard_negative_indices.append(idx)
                    if len(hard_negative_indices) >= n_hard_negatives:
                        break
            
            # Get hard negative texts
            hard_negative_texts = [self.corpus_texts[idx] for idx in hard_negative_indices]
            
            # Create examples with hard negatives for each positive
            for pos_text in positive_texts:
                if len(hard_negative_texts) > 0:
                    train_examples.append(InputExample(
                        texts=[query_text, pos_text] + hard_negative_texts[:n_hard_negatives]
                    ))
            
            # Add random negatives for diversity
            if n_random_negatives > 0:
                for pos_id in positive_ids:
                    pos_text = self.corpus_dict[pos_id]
                    
                    # Select random negatives
                    random_neg_indices = []
                    while len(random_neg_indices) < n_random_negatives:
                        idx = np.random.randint(0, len(self.corpus_texts))
                        if self.corpus_ids[idx] not in positive_ids and idx not in random_neg_indices:
                            random_neg_indices.append(idx)
                    
                    random_neg_texts = [self.corpus_texts[idx] for idx in random_neg_indices]
                    
                    train_examples.append(InputExample(
                        texts=[query_text, pos_text] + random_neg_texts
                    ))
        
        logger.info(f"Created {len(train_examples)} hard negative examples")
        return train_examples


class BaselineDataset(IRDatasetBase):
    """Dataset for baseline (no fine-tuning) strategy."""
    
    def create_train_examples(self) -> List[InputExample]:
        """
        Create empty training examples list (no fine-tuning).
        
        Returns:
            Empty list of InputExample objects
        """
        logger.info("Baseline strategy - no training examples created")
        return []


def get_dataset(strategy: str, data: Dict[str, Any], config: Dict[str, Any]) -> IRDatasetBase:
    """
    Factory function to get the appropriate dataset based on strategy.
    
    Args:
        strategy: Training strategy name
        data: Processed data dictionary
        config: Configuration dictionary
        
    Returns:
        Dataset instance
    """
    strategy_map = {
        "baseline": BaselineDataset,
        "positive_pair": PositivePairDataset,
        "hard_negative": HardNegativeDataset
    }
    
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(strategy_map.keys())}")
    
    return strategy_map[strategy](data, config)
