"""
Positive Pair trainer module.
"""

import logging
from typing import Dict, Any

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from viir.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PositivePairTrainer(BaseTrainer):
    """Trainer for positive pair fine-tuning strategy."""
    
    def train(self) -> SentenceTransformer:
        """
        Train the model using positive pairs.
        
        Returns:
            Fine-tuned SentenceTransformer model
        """
        logger.info("Positive pair strategy - creating training examples")
        
        # Create training examples
        train_examples = self.dataset.create_train_examples()
        
        # Skip training if no examples
        if len(train_examples) == 0:
            logger.warning("No training examples found, skipping training")
            return self.model
        
        # Log ví dụ đầu tiên để debug
        if len(train_examples) > 0:
            logger.info(f"Example training data - texts: {train_examples[0].texts}")
            logger.info(f"Example training data - label: {train_examples[0].label}")
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.config["training"]["batch_size"]
        )
        
        # Create loss function
        logger.info("Using CosineSimilarityLoss for positive pair training")
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create dev evaluator
        dev_evaluator = self.create_evaluator("dev")
        
        # Training parameters
        num_epochs = self.config["training"]["epochs"]
        evaluation_steps = self.config["training"]["evaluation_steps"]
        warmup_steps = self.config["training"].get("warmup_steps", 100)
        if "warmup_ratio" in self.config["training"] and len(train_examples) > 0:
            warmup_steps = int(len(train_examples) / self.config["training"]["batch_size"] * 
                              self.config["training"]["warmup_ratio"])
            logger.info(f"Setting warmup steps to {warmup_steps} based on warmup ratio")
        
        lr = self.config["training"]["learning_rate"]
        
        # Train the model
        logger.info(f"Training with {len(train_examples)} examples for {num_epochs} epochs")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': lr},
            output_path=f"{self.output_dir}/checkpoints"
        )
        
        # Save final model
        self.save_model()
        
        return self.model
