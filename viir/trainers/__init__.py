"""
Factory module for trainers.
"""

import logging
from typing import Dict, Any

from viir.data.dataset import IRDatasetBase
from viir.trainers.base_trainer import BaseTrainer
from viir.trainers.baseline_trainer import BaselineTrainer
from viir.trainers.positive_pair_trainer import PositivePairTrainer
from viir.trainers.hard_negative_trainer import HardNegativeTrainer

logger = logging.getLogger(__name__)


def get_trainer(strategy: str, dataset: IRDatasetBase, config: Dict[str, Any]) -> BaseTrainer:
    """
    Factory function to get the appropriate trainer based on strategy.
    
    Args:
        strategy: Training strategy name
        dataset: Dataset instance
        config: Configuration dictionary
        
    Returns:
        Trainer instance
    """
    strategy_map = {
        "baseline": BaselineTrainer,
        "positive_pair": PositivePairTrainer,
        "hard_negative": HardNegativeTrainer
    }
    
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(strategy_map.keys())}")
    
    logger.info(f"Using {strategy} training strategy")
    return strategy_map[strategy](dataset, config)