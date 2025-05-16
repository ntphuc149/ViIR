"""
Baseline trainer module (no fine-tuning).
"""

import logging
from typing import Dict, Any

from sentence_transformers import SentenceTransformer

from trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class BaselineTrainer(BaseTrainer):
    """Trainer for the baseline strategy (no fine-tuning)."""
    
    def train(self) -> SentenceTransformer:
        """
        No training for baseline, just evaluate the pre-trained model.
        
        Returns:
            Pre-trained SentenceTransformer model
        """
        logger.info("Baseline strategy - no training will be performed")
        
        # Create dev evaluator
        logger.info("Evaluating pre-trained model on dev set")
        dev_evaluator = self.create_evaluator("dev")
        
        # Evaluate on dev set
        dev_results = dev_evaluator(self.model)
        
        # Log results
        for metric, value in dev_results.items():
            logger.info(f"Dev {metric}: {value:.4f}")
        
        # Save model
        self.save_model()
        
        return self.model