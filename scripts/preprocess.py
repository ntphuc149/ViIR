"""
Script for data preprocessing.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viir.data.processor import DataProcessor
from viir.utils.logger import setup_logging
from viir.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ViBiDLQA data")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input CSV file")
    
    parser.add_argument("--output", "-o", type=str, default="data/processed",
                        help="Output directory for processed data")
    
    parser.add_argument("--config", "-c", type=str, default="viir/config/default.yaml",
                        help="Path to configuration file")
    
    parser.add_argument("--log", "-l", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point for preprocessing."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log.upper())
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output:
        config["data"]["output_dir"] = args.output
        logger.info(f"Output directory set to {args.output}")
    
    # Create output directory
    os.makedirs(config["data"]["output_dir"], exist_ok=True)
    
    # Process data
    logger.info(f"Processing data from {args.input}")
    processor = DataProcessor(config)
    data = processor.process(args.input)
    
    logger.info(f"Processed {len(data['corpus'])} documents and {len(data['queries'])} queries")
    logger.info(f"Data saved to {config['data']['output_dir']}")


if __name__ == "__main__":
    main()
