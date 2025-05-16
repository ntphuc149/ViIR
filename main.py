"""
Main entry point for ViBiDLQA-IR project.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

from data.processor import DataProcessor, load_processed_data
from data.dataset import get_dataset
from trainers import get_trainer
from evaluation.evaluator import IRModelEvaluator
from utils.logger import setup_logging
from utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViBiDLQA Information Retrieval")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input CSV file")
    
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="Base output directory")
    
    parser.add_argument("--strategy", "-s", type=str, default="hard_negative",
                        choices=["baseline", "positive_pair", "hard_negative"],
                        help="Training strategy")
    
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to configuration file (if not specified, uses the strategy-specific config)")
    
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip evaluation after training")
    
    parser.add_argument("--log", "-l", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.strategy}_{timestamp}")
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "vibidlqa-ir.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(level=args.log.upper(), log_file=log_file, console=True)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = f"config/{args.strategy}.yaml"
    
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Override configuration with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    # Update output directories
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "model")
    eval_dir = os.path.join(output_dir, "evaluation")
    
    config["data"]["output_dir"] = data_dir
    config["training"]["output_dir"] = model_dir
    config["evaluation"]["output_dir"] = eval_dir
    
    # Create output directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Log start message
    logger.info(f"Starting ViBiDLQA-IR with strategy: {args.strategy}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process data
    logger.info("Processing data")
    processor = DataProcessor(config)
    data = processor.process(args.input)
    
    # Create dataset
    logger.info("Creating dataset")
    dataset = get_dataset(args.strategy, data, config)
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = get_trainer(args.strategy, dataset, config)
    
    # Train model
    logger.info("Starting training")
    start_time = time.time()
    model = trainer.train()
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    if not args.no_eval:
        logger.info("Evaluating model")
        evaluator = IRModelEvaluator(model, data, config)
        results = evaluator.evaluate_all_splits()
        
        # Print test results
        logger.info("Test results:")
        for metric, value in results["test"].items():
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"All operations completed successfully. Results saved to {output_dir}")


if __name__ == "__main__":
    main()