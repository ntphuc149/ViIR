"""
Script for model training.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

import yaml

from data.processor import load_processed_data
from data.dataset import get_dataset
from trainers import get_trainer
from utils.logger import setup_logging
from utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IR model for ViBiDLQA")
    
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to configuration file")
    
    parser.add_argument("--data_dir", "-d", type=str, default=None,
                        help="Directory containing processed data")
    
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for model and results")
    
    parser.add_argument("--log", "-l", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log.upper())
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override data directory if specified
    if args.data_dir:
        config["data"]["output_dir"] = args.data_dir
        logger.info(f"Data directory set to {args.data_dir}")
    
    # Override output directory if specified
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
        config["evaluation"]["output_dir"] = os.path.join(args.output_dir, "evaluation")
        logger.info(f"Output directory set to {args.output_dir}")
    
    # Create output directories
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    os.makedirs(config["evaluation"]["output_dir"], exist_ok=True)
    
    # Load processed data
    logger.info(f"Loading processed data from {config['data']['output_dir']}")
    data = load_processed_data(config["data"]["output_dir"])
    
    # Get strategy
    strategy = config.get("strategy", "hard_negative")
    logger.info(f"Using {strategy} training strategy")
    
    # Create dataset
    logger.info("Creating dataset")
    dataset = get_dataset(strategy, data, config)
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = get_trainer(strategy, dataset, config)
    
    # Train model
    logger.info("Starting training")
    model = trainer.train()
    
    # Save model
    logger.info(f"Training complete. Model saved to {config['training']['output_dir']}")
    
    # Evaluate on test set
    from evaluation.evaluator import IRModelEvaluator
    logger.info("Evaluating model on test set")
    evaluator = IRModelEvaluator(model, data, config)
    results = evaluator.evaluate("test")
    
    logger.info("Training and evaluation complete")


if __name__ == "__main__":
    main()