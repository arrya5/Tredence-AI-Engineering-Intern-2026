import logging
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Project configuration using Pydantic for type safety and environment variable parsing.
    """
    # Model parameters
    input_size: int = 3072  # 3 * 32 * 32 (CIFAR-10 flattened)
    hidden_size_1: int = 512
    hidden_size_2: int = 256
    num_classes: int = 10
    
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 0.003
    epochs: int = 20  # Boosted slightly to ensure L1 regularizer has time to flatten gates
    
    # Regularization (We will iterate over these in train.py)
    lambdas: list[float] = [0.0001, 0.01, 0.5]
    sparsity_threshold: float = 1e-2
    
    # Logging configuration
    log_level: str = "INFO"

    class Config:
        env_prefix = "TREDENCE_"

# Instantiate settings
config = Settings()

# Setup standard logger
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TredenceAI")
