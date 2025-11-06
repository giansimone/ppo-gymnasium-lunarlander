"""
Module for utility functions and configuration file handling.
"""
import random
from pathlib import Path

import torch
import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(filepath: Path) -> dict:
    """Load the YAML configuration file into a standard Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, filepath: Path) -> None:
    """Save a standard Python dictionary into a YAML configuration file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
