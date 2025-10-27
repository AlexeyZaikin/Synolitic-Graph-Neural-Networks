import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class GNNTrainer:
    """Optimized GNN trainer with automatic configuration"""

    def __init__(self, cfg: DictConfig, device: str = "mps"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.metric_history = defaultdict(list)

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Efficient training loop with autocast"""
        model.train()
        total_loss = 0
        all_probs, all_labels, dataset_names = [], [], []

        for batch in loader:
            batch_device = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with torch.autocast(
                enabled=self.cfg.training.mixed_precision, device_type=self.device.type
            ):
                out = model(batch_device)
                loss = criterion(out, batch_device.y.long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_device.num_graphs
            probs = F.softmax(out, dim=1).detach().cpu()
            all_probs.append(probs)
            all_labels.append(batch_device.y.cpu())
            dataset_names.append(batch_device.dataset_name)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        return total_loss / len(loader.dataset), all_probs, all_labels  # type: ignore

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, loader: DataLoader, criterion: nn.Module
    ) -> dict[str, float]:
        """Comprehensive model evaluation with per-dataset metrics"""
        model.eval()
        total_loss = 0
        all_probs, all_preds, all_labels = [], [], []
        all_dataset_names = []  # Track dataset names

        for batch in loader:
            batch_device = batch.to(self.device)
            out = model(batch_device)
            loss = criterion(out, batch_device.y.long())

            total_loss += loss.item() * batch_device.num_graphs
            probs = F.softmax(out, dim=1).detach().cpu()
            preds = probs.argmax(dim=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(batch_device.y.cpu())
            all_dataset_names.extend(batch_device.dataset_name)  # Collect dataset names

        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute global metrics
        global_metrics = self._compute_metrics(all_probs, all_preds, all_labels)
        metrics = {
            **global_metrics,
            "loss": total_loss / len(loader.dataset),  # type: ignore
            "global": global_metrics,
            "per_dataset": {},
        }

        # Compute per-dataset metrics
        unique_datasets = set(all_dataset_names)
        for dataset in unique_datasets:
            mask = [ds == dataset for ds in all_dataset_names]
            dataset_probs = all_probs[mask]
            dataset_preds = all_preds[mask]
            dataset_labels = all_labels[mask]

            if len(dataset_probs) > 0:  # Only compute if data exists
                dataset_metrics = self._compute_metrics(
                    dataset_probs, dataset_preds, dataset_labels
                )
                metrics["per_dataset"][dataset] = dataset_metrics

        return metrics

    @classmethod
    def _compute_metrics(
        cls, probs: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        labels_np = labels.numpy()
        probs_np = probs.numpy()
        preds_np = preds.numpy()

        metrics = {
            "accuracy": (preds_np == labels_np).mean(),
            "precision": precision_score(
                labels_np, preds_np, average="binary", pos_label=1, zero_division=0
            ),
            "recall": recall_score(
                labels_np, preds_np, average="binary", pos_label=1, zero_division=0
            ),
            "f1": f1_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0),
        }

        # ROC-AUC calculation
        metrics["roc_auc"] = roc_auc_score(labels_np, probs_np[:, 1])

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(labels_np, probs_np[:, 1])
        metrics["pr_auc"] = auc(recall, precision)

        return metrics

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        logger: logging.Logger,
        tb_writer: SummaryWriter,
        log_dir: Path,
    ) -> tuple[dict[str, list], nn.Module]:
        """Training loop with early stopping and checkpointing"""
        best_val_roc_auc = -999
        early_stop_counter = 0
        history = defaultdict(list)
        epochs = self.cfg.training.max_epochs
        patience = self.cfg.training.patience

        for epoch in tqdm(range(1, epochs + 1), desc="Training model"):
            start_time = time.time()

            # Training phase
            train_loss, train_probs, train_labels = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            train_metrics = self._compute_metrics(
                train_probs, train_probs.argmax(dim=1), train_labels
            )

            # Validation phase
            val_metrics = self.evaluate(model, val_loader, criterion)

            # Update learning rate scheduler
            scheduler.step(epoch)

            # Track history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_roc_auc"].append(train_metrics["roc_auc"])
            history["train_f1"].append(train_metrics["f1"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_roc_auc"].append(val_metrics["roc_auc"])
            history["val_f1"].append(val_metrics["f1"])
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)

            # TensorBoard logging
            tb_writer.add_scalar("train/Loss", train_loss, epoch)
            tb_writer.add_scalar("val/Loss", val_metrics["loss"], epoch)
            tb_writer.add_scalar("val/ROC-AUC", val_metrics["roc_auc"], epoch)
            tb_writer.add_scalar("val/Accuracy", val_metrics["accuracy"], epoch)
            tb_writer.add_scalar("val/F1", val_metrics["f1"], epoch)
            tb_writer.add_scalar("train/Learning Rate", history["lr"][-1], epoch)

            # Checkpointing
            if val_metrics["roc_auc"] > best_val_roc_auc:
                best_val_roc_auc = val_metrics["roc_auc"]
                early_stop_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_roc_auc": best_val_roc_auc,
                    },
                    log_dir / "best_model.pth",
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress
            if epoch % self.cfg.training.log_interval == 0:
                epoch_time = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | "
                    f"LR: {history['lr'][-1]:.6f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
                )
        # Load best model
        checkpoint = torch.load(log_dir / "best_model.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        return history, model
