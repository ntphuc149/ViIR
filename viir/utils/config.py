"""
Configuration utilities.
"""

import os
from typing import Dict, Any, List

import yaml


def find_config_file(config_path: str) -> str:
    """
    Find the config file by searching in multiple locations.
    
    Args:
        config_path: Base path to the YAML configuration file
        
    Returns:
        Resolved path to the configuration file
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found
    """
    search_paths = [
        config_path,                            # Direct path
        os.path.join("viir", config_path),      # Inside viir directory
        os.path.join("viir", "config", os.path.basename(config_path))  # Inside viir/config
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    # If we reach here, the file was not found
    raise FileNotFoundError(f"Config file not found: {config_path}. Searched in: {search_paths}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    resolved_path = find_config_file(config_path)
    
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Process inheritance
    if "inherit" in config:
        parent_path = config.pop("inherit")
        if not os.path.isabs(parent_path):
            # Make parent path relative to current config file
            config_dir = os.path.dirname(resolved_path)
            parent_path = os.path.join(config_dir, os.path.basename(parent_path))
        
        try:
            parent_config = load_config(parent_path)
            
            # Merge configs: parent values are overridden by child values
            merged_config = _deep_merge(parent_config, config)
            return merged_config
        except FileNotFoundError:
            # If parent config not found, just use current config
            print(f"Warning: Parent config {parent_path} not found, using only child config")
            return config
    
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
