"""
Hard Negative trainer module.
"""

import logging
from typing import Dict, Any

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from viir.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class HardNegativeTrainer(BaseTrainer):
    """Trainer for hard negative fine-tuning strategy."""
    
    def train(self) -> SentenceTransformer:
        """
        Train the model using hard negatives.
        
        Returns:
            Fine-tuned SentenceTransformer model
        """
        logger.info("Hard negative strategy - creating training examples")
        
        # Create training examples with hard negatives
        train_examples = self.dataset.create_train_examples()
        
        # Skip training if no examples
        if len(train_examples) == 0:
            logger.warning("No training examples found, skipping training")
            return self.model
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.config["training"]["batch_size"]
        )
        
        # Create loss function
        logger.info("Using MultipleNegativesRankingLoss for hard negative training")
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Create dev evaluator
        dev_evaluator = self.create_evaluator("dev")
        
        # Training parameters
        num_epochs = self.config["training"]["epochs"]
        evaluation_steps = self.config["training"]["evaluation_steps"]
        warmup_steps = self.config["training"]["warmup_steps"]
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