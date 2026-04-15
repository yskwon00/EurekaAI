"""EurekaAI — Core package."""
from .model.architecture import EurekaModel
from .model.config import EurekaConfig, tiny_config, small_config, medium_config
from .model.tokenizer_utils import EurekaTokenizer

__version__ = "0.1.0"
__all__ = [
    "EurekaModel",
    "EurekaConfig",
    "EurekaTokenizer",
    "tiny_config",
    "small_config",
    "medium_config",
]
