import json
import os
import pickle
import traceback
import warnings
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.trial import Trial
from sgnn.model import GNNModel
from sgnn.node_features_utils import add_node_features
from sgnn.sparsify_utils import get_sparsify_f_list
from sgnn.trainer import GNNTrainer
from sgnn.utils import cleanup_logging, plot_metrics, setup_logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def objective(trial: Trial, cfg: DictConfig, full_data: list[Data], log_dir: Path) -> float:  # noqa
    """Optuna hyperparameter optimization objective"""
    logger = None
    tb_writer = None

    try:
        # Suggest hyperparameters
        params = {
            "model": {
                "activation": trial.suggest_categorical(
                    "activation", cfg.hparams.activation.options
                ),
                "hidden_channels": trial.suggest_int(
                    "hidden_channels",
                    cfg.hparams.hidden_channels.min,
                    cfg.hparams.hidden_channels.max,
                ),
                "num_layers": trial.suggest_int(
                    "num_layers", cfg.hparams.num_layers.min, cfg.hparams.num_layers.max
                ),
                "dropout": trial.suggest_float(
                    "dropout", cfg.hparams.dropout.min, cfg.hparams.dropout.max
                ),
                "residual": trial.suggest_categorical("residual", cfg.hparams.residual.options),
                "use_classifier_mlp": trial.suggest_categorical(
                    "use_classifier_mlp", cfg.hparams.use_classifier_mlp.options
                ),
            },
            "training": {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    cfg.hparams.learning_rate.min,
                    cfg.hparams.learning_rate.max,
                    log=True,
                ),
            },
        }

        if cfg.model.type == "GCN":
            params["model"]["use_edge_encoder"] = False
        else:
            params["model"]["use_edge_encoder"] = trial.suggest_categorical(
                "use_edge_encoder", cfg.hparams.use_edge_encoder.options
            )

        if cfg.model.type in {"GATv2", "Transformer"}:
            params["model"]["heads"] = trial.suggest_int(
                "heads", cfg.hparams.heads.min, cfg.hparams.heads.max
            )
            params["model"]["concat"] = trial.suggest_categorical(
                "concat", cfg.hparams.concat.options
            )

        if params["model"]["use_edge_encoder"]:
            params["model"]["edge_encoder_channels"] = trial.suggest_int(
                "edge_encoder_channels",
                cfg.hparams.edge_encoder_channels.min,
                cfg.hparams.edge_encoder_channels.max,
            )
            params["model"]["edge_encoder_layers"] = trial.suggest_int(
                "edge_encoder_layers",
                cfg.hparams.edge_encoder_layers.min,
                cfg.hparams.edge_encoder_layers.max,
            )

        if params["model"]["use_classifier_mlp"]:
            params["model"]["classifier_mlp_channels"] = trial.suggest_int(
                "classifier_mlp_channels",
                cfg.hparams.classifier_mlp_channels.min,
                cfg.hparams.classifier_mlp_channels.max,
            )
            params["model"]["classifier_mlp_layers"] = trial.suggest_int(
                "classifier_mlp_layers",
                cfg.hparams.classifier_mlp_layers.min,
                cfg.hparams.classifier_mlp_layers.max,
            )

        # Create trial-specific config
        trial_cfg = OmegaConf.merge(cfg, OmegaConf.create(params))

        # Set up logging
        trial_log_dir = log_dir / f"trial_{trial.number}"
        trial_log_dir.mkdir(exist_ok=True)
        logger, tb_writer = setup_logging(trial_log_dir)

        logger.info(f"Current trial cfg: {trial_cfg}")

        if cfg.use_kfold:
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(
                n_splits=trial_cfg.training.cv_folds,
                shuffle=True,
                random_state=trial_cfg.seed,
            )
            labels = [data.y for data in full_data]

            for fold, (train_idx, val_idx) in tqdm(
                enumerate(skf.split(full_data, labels)), desc="CV Fold"
            ):
                fold_log_dir = trial_log_dir / f"fold_{fold}"
                fold_log_dir.mkdir(exist_ok=True)

                # Create data loaders
                train_loader = DataLoader(
                    [full_data[i] for i in train_idx],
                    shuffle=True,
                    batch_size=trial_cfg.training.batch_size,
                    num_workers=0,
                    pin_memory=True,
                )
                val_loader = DataLoader(
                    [full_data[i] for i in val_idx],
                    batch_size=trial_cfg.training.batch_size,
                    num_workers=0,
                    pin_memory=True,
                )

                # Initialize model
                model = GNNModel(
                    trial_cfg,
                    in_channels=full_data[0].x.shape[-1],
                ).to(torch.device(trial_cfg.device))

                # Optimizer and scheduler
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=trial_cfg.training.learning_rate,
                    weight_decay=trial_cfg.training.weight_decay,
                )
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    patience=trial_cfg.training.lr_patience,
                    factor=trial_cfg.training.lr_factor,
                )
                criterion = nn.CrossEntropyLoss()

                # Trainer setup
                trainer = GNNTrainer(trial_cfg, device=trial_cfg.device)

                try:
                    # Training
                    history, model = trainer.train(
                        model,
                        train_loader,
                        val_loader,
                        optimizer,
                        criterion,
                        scheduler,
                        logger,
                        tb_writer,
                        fold_log_dir,
                    )
                    logger.info(
                        f"Finished training for fold {fold}. Current history is being saved."
                    )
                    with Path(fold_log_dir / "history.json").open("w", encoding="utf-8") as f:
                        json.dump(history, f)

                    # Validation metrics
                    val_metrics = trainer.evaluate(model, val_loader, criterion)
                    cv_scores.append(val_metrics["roc_auc"])

                    # Report intermediate result
                    trial.report(val_metrics["roc_auc"], fold)

                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.TrialPruned

                except Exception:
                    logger.error(f"Training failed: {traceback.format_exc()}")
                    cv_scores.append(0.0)

            return np.mean(cv_scores)
        fold = 0
        fold_log_dir = trial_log_dir / f"fold_{fold}"
        fold_log_dir.mkdir(exist_ok=True)
        labels = [data.y for data in full_data]
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)), train_size=0.9, stratify=labels, random_state=trial_cfg.seed
        )
        # Create data loaders
        train_loader = DataLoader(
            [full_data[i] for i in train_idx],
            shuffle=True,
            batch_size=trial_cfg.training.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            [full_data[i] for i in val_idx],
            batch_size=trial_cfg.training.batch_size,
            num_workers=0,
            pin_memory=True,
        )

        # Initialize model
        model = GNNModel(
            trial_cfg,
            in_channels=full_data[0].x.shape[-1],
        ).to(torch.device(trial_cfg.device))

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=trial_cfg.training.lr_patience,
            factor=trial_cfg.training.lr_factor,
        )
        criterion = nn.CrossEntropyLoss()

        # Trainer setup
        trainer = GNNTrainer(trial_cfg, device=trial_cfg.device)

        try:
            # Training
            history, model = trainer.train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                scheduler,
                logger,
                tb_writer,
                fold_log_dir,
            )
            logger.info("Finished training. Current history is being saved.")
            with Path(fold_log_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f)

            # Validation metrics
            val_metrics = trainer.evaluate(model, val_loader, criterion)
            score = val_metrics["roc_auc"]

            # Report intermediate result
            trial.report(val_metrics["roc_auc"], fold)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned

        except Exception:
            logger.error(f"Training failed: {traceback.format_exc()}")
            score = 0.0

        return score

    finally:
        # Always cleanup resources, even if an exception occurred
        if logger is not None and tb_writer is not None:
            cleanup_logging(logger, tb_writer)


def main_loop(cfg: DictConfig, selected_data, base_dir):
    # sparsify and add node features
    p_list = [0.2, 0.8]
    sparsify_functions_list = get_sparsify_f_list(p_list)
    for trial_idx, (sparsify_tuple, node_features) in enumerate(
        product(sparsify_functions_list, [True, False])
    ):
        logger = None
        tb_writer = None

        try:
            sparsify_name = sparsify_tuple[0]
            sparsify_f = sparsify_tuple[1]
            data = sparsify_f(selected_data)
            if node_features:
                data = add_node_features(data)

            cfg.data.sparsify = sparsify_name
            cfg.data.node_features = node_features
            cfg.data.trial_idx = trial_idx

            # Prepare data
            train_data = data["train"]
            test_data = data["test"]
            full_data = train_data  # For cross-validation
            model_type = cfg.model.type

            # Split train_data into train and validation
            train_data_split, val_data_split = train_test_split(
                train_data,
                test_size=0.1,
                random_state=cfg.seed,
                stratify=[int(item.y) for item in train_data],
            )

            model_dir = (
                base_dir
                / model_type
                / cfg.data.sparsify
                / f"node_features_{cfg.data.node_features}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            # Set up logging
            logger, tb_writer = setup_logging(model_dir)
            logger.info(
                f"Starting experiment: {model_type}/{cfg.data.sparsify}/node_features_{cfg.data.node_features}"
            )

            if cfg.optimize:
                # Optuna hyperparameter optimization
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=cfg.optuna.n_startup_trials,
                        n_warmup_steps=cfg.optuna.n_warmup_steps,
                    ),
                )

                study.optimize(
                    lambda trial: objective(trial, cfg, full_data, model_dir),
                    n_trials=cfg.optuna.n_trials,
                    timeout=cfg.optuna.timeout,
                    show_progress_bar=True,
                )

                # Save best parameters
                best_params = study.best_params
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best ROC-AUC: {study.best_value:.4f}")

                # Final training with best parameters
                cfg.model.update(best_params.get("model", {}))
                cfg.training.update(best_params.get("training", {}))

            # Data loaders
            train_loader = DataLoader(
                train_data_split,
                shuffle=True,
                batch_size=cfg.training.batch_size,
                num_workers=0,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_data_split,
                batch_size=cfg.training.batch_size,
                num_workers=0,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_data,
                batch_size=cfg.training.batch_size,
                num_workers=0,
                pin_memory=True,
            )

            # Initialize model
            model = GNNModel(
                cfg,
                in_channels=train_data[0].x.shape[-1],
            ).to(torch.device(cfg.device))

            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=cfg.training.lr_patience,
                factor=cfg.training.lr_factor,
            )
            criterion = nn.CrossEntropyLoss()

            try:
                # Train final model
                trainer = GNNTrainer(cfg, device=cfg.device)
                history, model = trainer.train(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    scheduler,
                    logger,
                    tb_writer,
                    model_dir,
                )

                # Final evaluation
                test_metrics = trainer.evaluate(model, test_loader, criterion)

                # Log results
                logger.info(f"\n{'=' * 50}")
                logger.info(f"FINAL RESULTS: model_type={model_type}")
                logger.info(f"Global Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
                logger.info(f"Global Test F1: {test_metrics['f1']:.4f}")
                # Log per-dataset metrics
                logger.info("\nPer-dataset metrics:")
                dataset_metrics = sorted(
                    test_metrics["per_dataset"].items(),
                    key=lambda x: x[1]["roc_auc"],
                    reverse=True,
                )
                for dataset, metrics_dict in dataset_metrics:
                    logger.info(
                        f"{dataset}: ROC-AUC={metrics_dict['roc_auc']:.4f} | "
                        f"F1={metrics_dict['f1']:.4f} | "
                        f"Accuracy={metrics_dict['accuracy']:.4f}"
                    )

                logger.info("=" * 50)

                # Visualizations
                plot_metrics(history, model_dir)
            except Exception:
                if logger:
                    logger.error(
                        f"Error in {model_type} during final model training:\n{traceback.format_exc()}",
                    )
        except Exception:
            if logger:
                logger.error(
                    f"Error in {model_type}:\n{traceback.format_exc()}",
                )
        finally:
            # Always cleanup resources, even if an exception occurred
            if logger is not None and tb_writer is not None:
                cleanup_logging(logger, tb_writer)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main experiment runner with Hydra configuration"""
    # Initialize output directory
    base_dir = Path(cfg.save_path) / "logs" / str(cfg.data.dataset_size)

    # Load datasets
    dataset_path = os.path.join(
        cfg.data.dataset_path,
        f"csv_{cfg.data.dataset_size}",
        "noisy" if cfg.expand_features else "",
        "processed_graphs.pkl",
    )
    dataset_names = cfg.data.datasets
    with Path(dataset_path).open("rb") as f:
        all_data = pickle.load(f)
    if cfg.per_dataset:
        if not dataset_names:
            dataset_names = list(all_data.keys())
        for dataset_name in dataset_names:
            cur_dir = base_dir / dataset_name
            selected_data = all_data[dataset_name]
            main_loop(cfg, selected_data, cur_dir)
    elif cfg.leave_one_out:
        if not dataset_names:
            dataset_names = list(all_data.keys())
        for test_dataset in tqdm(dataset_names):
            cur_dir = base_dir / f"leave_one_out_{test_dataset}"
            train_data = []
            for name in dataset_names:
                if name != test_dataset:
                    train_data.extend(all_data[name]["train"])
                    train_data.extend(all_data[name]["test"])
            test_data = all_data[test_dataset]["test"]
            selected_data = {"train": train_data, "test": test_data}
            main_loop(cfg, selected_data, cur_dir)
    else:
        selected_data = {"train": [], "test": []}
        if not dataset_names:
            dataset_names = list(all_data.keys())
            for dataset_name in tqdm(dataset_names, desc="Loading datasets"):
                data = all_data[dataset_name]
                selected_data["train"].extend(data["train"])
                selected_data["test"].extend(data["test"])
            main_loop(cfg, selected_data, base_dir)


if __name__ == "__main__":
    main()
