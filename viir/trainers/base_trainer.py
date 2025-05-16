"""
Base trainer module.
"""

import logging
import os
from typing import Dict, List, Tuple, Any, Optional

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

from viir.data.dataset import IRDatasetBase

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, dataset: IRDatasetBase, config: Dict[str, Any]):
        """
        Initialize the base trainer.
        
        Args:
            dataset: Dataset instance
            config: Configuration dictionary
        """
        self.dataset = dataset
        self.config = config
        self.model_name = config["model"]["name"]
        self.output_dir = config["training"]["output_dir"]
        self.device = config["training"]["device"]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        logger.info(f"Initializing model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name, 
            trust_remote_code=True,
            device=self.device
        )
        
        if "max_seq_length" in config["model"]:
            self.model.max_seq_length = config["model"]["max_seq_length"]
    
    def train(self) -> SentenceTransformer:
        """
        Train the model.
        
        Returns:
            Trained SentenceTransformer model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_evaluator(self, split: str) -> evaluation.InformationRetrievalEvaluator:
        """
        Create an evaluator for the specified split.
        
        Args:
            split: Data split to evaluate on ("train", "dev", or "test")
            
        Returns:
            InformationRetrievalEvaluator for the specified split
        """
        # Get evaluation data
        queries, corpus, relevant_docs = self.dataset.create_eval_data(split)
        
        # Create evaluator
        evaluator = evaluation.InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"{split}-evaluation",
            batch_size=self.config["evaluation"]["batch_size"],
            show_progress_bar=True
        )
        
        return evaluator
    
    def save_model(self) -> None:
        """Save the model to the output directory."""
        model_save_path = os.path.join(self.output_dir, "model")
        logger.info(f"Saving model to {model_save_path}")
        self.model.save(model_save_path)