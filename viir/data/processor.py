"""
Data processor module for ViBiDLQA dataset.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processor for ViBiDLQA dataset.
    
    This class handles reading, processing, and splitting the dataset into 
    corpus, queries, and relevance judgments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary with data processing parameters
        """
        self.config = config
        self.output_dir = config["data"]["output_dir"]
        self.train_ratio = config["data"]["train_ratio"]
        self.dev_ratio = config["data"]["dev_ratio"]
        self.test_ratio = config["data"]["test_ratio"]
        self.seed = config["data"]["seed"]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
    
    def process(self, csv_path: str) -> Dict[str, Any]:
        """
        Process the ViBiDLQA dataset from CSV.
        
        Args:
            csv_path: Path to the CSV file containing the dataset
            
        Returns:
            Dictionary containing processed data:
            - corpus: List of documents with IDs and text
            - queries: List of queries with IDs and text
            - qrels: List of query-document relevance judgments
            - splits: Dictionary with train/dev/test query IDs
            - context_to_id: Mapping from context text to document ID
        """
        logger.info(f"Processing data from {csv_path}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        # Create unique context mapping
        unique_contexts = df['context'].unique()
        context_to_id = {context: f"doc_{i}" for i, context in enumerate(unique_contexts)}
        logger.info(f"Found {len(unique_contexts)} unique contexts")
        
        # Create corpus
        corpus = [{"id": doc_id, "text": context} for context, doc_id in context_to_id.items()]
        self._save_jsonl(corpus, os.path.join(self.output_dir, "corpus.jsonl"))
        
        # Create mapping from context to questions
        context_to_questions = {}
        for _, row in df.iterrows():
            ctx = row['context']
            if ctx not in context_to_questions:
                context_to_questions[ctx] = []
            context_to_questions[ctx].append({
                "question": row['question'], 
                "answer": row['abstractive_answer']
            })
        
        # Create queries and qrels
        queries = []
        qrels = []
        
        for context, questions in context_to_questions.items():
            doc_id = context_to_id[context]
            for q in questions:
                query_id = f"query_{len(queries)}"
                queries.append({"id": query_id, "text": q['question']})
                qrels.append({"query_id": query_id, "doc_id": doc_id, "score": 1})
        
        # Save queries
        self._save_jsonl(queries, os.path.join(self.output_dir, "queries.jsonl"))
        
        # Save qrels
        self._save_jsonl(qrels, os.path.join(self.output_dir, "qrels.jsonl"))
        
        # Split data into train/dev/test
        splits = self._create_splits(queries)
        
        # Save splits
        with open(os.path.join(self.output_dir, "splits.json"), "w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(unique_contexts)} unique contexts, {len(queries)} queries")
        logger.info(f"Train: {len(splits['train'])}, Dev: {len(splits['dev'])}, Test: {len(splits['test'])}")
        
        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
            "splits": splits,
            "context_to_id": context_to_id
        }
    
    def _create_splits(self, queries: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Split query IDs into train, dev, and test sets.
        
        Args:
            queries: List of query dictionaries with IDs
            
        Returns:
            Dictionary with train, dev, and test query ID lists
        """
        n_queries = len(queries)
        indices = np.random.permutation(n_queries)
        
        train_size = int(self.train_ratio * n_queries)
        dev_size = int(self.dev_ratio * n_queries)
        
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:train_size+dev_size]
        test_indices = indices[train_size+dev_size:]
        
        return {
            "train": [queries[i]["id"] for i in train_indices],
            "dev": [queries[i]["id"] for i in dev_indices],
            "test": [queries[i]["id"] for i in test_indices]
        }
    
    def _save_jsonl(self, data: List[Dict], output_path: str) -> None:
        """
        Save data as JSONL file.
        
        Args:
            data: List of dictionaries to save
            output_path: Path to save the JSONL file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(data)} items to {output_path}")


def load_processed_data(data_dir: str) -> Dict[str, Any]:
    """
    Load pre-processed data from directory.
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        Dictionary with loaded data
    """
    logger.info(f"Loading processed data from {data_dir}")
    
    # Load corpus
    corpus = []
    with open(os.path.join(data_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    
    # Load queries
    queries = []
    with open(os.path.join(data_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    
    # Load qrels
    qrels = []
    with open(os.path.join(data_dir, "qrels.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))
    
    # Load splits
    with open(os.path.join(data_dir, "splits.json"), "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Create context_to_id mapping
    context_to_id = {doc["text"]: doc["id"] for doc in corpus}
    
    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
    
    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "splits": splits,
        "context_to_id": context_to_id
    }