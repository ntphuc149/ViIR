"""
Script for model evaluation.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

import yaml

from viir.data.processor import load_processed_data
from viir.evaluation.evaluator import evaluate_model
from viir.utils.logger import setup_logging
from viir.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate IR model for ViBiDLQA")
    
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to the trained model")
    
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                        help="Directory containing processed data")
    
    parser.add_argument("--config", "-c", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for evaluation results")
    
    parser.add_argument("--split", "-s", type=str, default="test",
                        choices=["train", "dev", "test", "all"],
                        help="Data split to evaluate on")
    
    parser.add_argument("--log", "-l", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point for evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log.upper())
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config["evaluation"]["output_dir"] = args.output_dir
        logger.info(f"Evaluation output directory set to {args.output_dir}")
    
    # Create output directory
    os.makedirs(config["evaluation"]["output_dir"], exist_ok=True)
    
    # Load data
    logger.info(f"Loading processed data from {args.data_dir}")
    data = load_processed_data(args.data_dir)
    
    # Load model and evaluate
    logger.info(f"Loading model from {args.model_path}")
    
    if args.split == "all":
        # Evaluate on all splits
        results = evaluate_model(args.model_path, args.data_dir, config)
        
        # Print summary
        logger.info("Evaluation results:")
        for split, split_results in results.items():
            logger.info(f"--- {split.capitalize()} ---")
            for metric, value in split_results.items():
                logger.info(f"{metric}: {value:.4f}")
    else:
        # Evaluate on specified split
        from sentence_transformers import SentenceTransformer
        from viir.evaluation.evaluator import IRModelEvaluator
        
        model = SentenceTransformer(args.model_path)
        evaluator = IRModelEvaluator(model, data, config)
        results = evaluator.evaluate(args.split)
        
        # Print results
        logger.info(f"Evaluation results ({args.split}):")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()