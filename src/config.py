"""Configuration management for MotionDetector settings."""
import json
import os
from typing import Dict, Any, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "min_area_pixels": 500,
    "min_area_percent": 0.5,
    "sections": "1_2_3_4",
    "cooldown_seconds": 1.0,
    "model": "yolov8n.pt",
    "confidence_threshold": 0.35,
}

CONFIG_FILE = "config.json"


def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from JSON file, returning defaults if file doesn't exist."""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
    return config


def save_config(config_data: Dict[str, Any], config_path: str = CONFIG_FILE) -> bool:
    """Save configuration dictionary to a JSON file."""
    try:
        # Merge with existing
        existing = load_config(config_path)
        existing.update(config_data)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config to {config_path}: {e}")
        return False
