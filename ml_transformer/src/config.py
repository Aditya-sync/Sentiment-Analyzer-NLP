"""
Configuration file for Transformer V2 sentiment analysis pipeline.
Ensures reproducibility, clean path resolution, and consistent experiment parameters.
"""

from pathlib import Path

# ============================================================================
# Paths Configuration
# ============================================================================
SRC_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SRC_DIR.parent
PROJECT_ROOT = MODULE_ROOT.parent

DATA_DIR = MODULE_ROOT / "data"
MODELS_DIR = MODULE_ROOT / "models"
OUTPUTS_DIR = MODULE_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS_DIR

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"

# ============================================================================
# Model & Tokenizer Configuration
# ============================================================================
MODEL_NAME = "FacebookAI/roberta-base"
NUM_LABELS = 3
MAX_LENGTH = 128

# Label Mappings (0 = negative, 1 = neutral, 2 = positive)
ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# ============================================================================
# Training Hyperparameters (Optimized for GPU Fine-Tuning)
# ============================================================================
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_RATIO = 0.1
METRIC_FOR_BEST_MODEL = "macro_f1"

# Target directory for saved fine-tuned model checkpoint
SAVED_MODEL_DIR = MODELS_DIR / "roberta-tweeteval"

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set random seeds across Python, NumPy, PyTorch, and Transformers for reproducible runs."""
    import random
    import numpy as np
    import torch
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
