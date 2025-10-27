import gc
import logging
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")


# Set global seeds for reproducibility
def set_global_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.default_rng(seed=seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_global_seed()


def plot_metrics(history: dict[str, list], log_dir: Path) -> None:
    """Create comprehensive metric visualizations"""
    plt.figure(figsize=(12, 8))

    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], "b-", label="Train")
    plt.plot(history["epoch"], history["val_loss"], "r-", label="Validation")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ROC-AUC
    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["val_roc_auc"], "g-")
    plt.title("Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")

    # F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["val_f1"], "m-")
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")

    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["val_acc"], "c-")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(log_dir / "training_metrics.png")
    plt.close()

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / "training_history.csv", index=False)


def setup_logging(log_dir: Path) -> tuple[logging.Logger, SummaryWriter]:
    """Comprehensive logging setup with unique logger per experiment"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create unique logger name based on log directory path
    logger_name = f"gnn_experiment_{hash(str(log_dir))}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent accumulation
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if hasattr(handler, "close"):
            handler.close()

    # File handler
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=log_dir)

    return logger, tb_writer


def cleanup_logging(logger: logging.Logger, tb_writer: SummaryWriter) -> None:
    """Properly cleanup logging resources"""
    try:
        # Close all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if hasattr(handler, "close"):
                handler.close()

        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()

        # Force garbage collection

        gc.collect()
    except Exception as e:
        print(f"Warning: Error during logging cleanup: {e}")
