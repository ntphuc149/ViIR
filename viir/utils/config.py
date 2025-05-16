"""
Configuration utilities.
"""

import os
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Process inheritance
    if "inherit" in config:
        parent_path = config.pop("inherit")
        if not os.path.isabs(parent_path):
            # Make parent path relative to current config file
            config_dir = os.path.dirname(config_path)
            parent_path = os.path.join(config_dir, parent_path)
        
        parent_config = load_config(parent_path)
        
        # Merge configs: parent values are overridden by child values
        merged_config = _deep_merge(parent_config, config)
        return merged_config
    
    return config


def _deep_merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Child values override parent values. For nested dictionaries, merges recursively.
    
    Args:
        parent: Parent dictionary
        child: Child dictionary (overrides parent values)
        
    Returns:
        Merged dictionary
    """
    result = parent.copy()
    
    for key, value in child.items():
        if key in parent and isinstance(parent[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(parent[key], value)
        else:
            # Override or add the value
            result[key] = value
    
    return result