import csv
import json
import logging
import pickle  # noqa: S403
import time
import traceback
import warnings
from ctypes import ArgumentError
from pathlib import Path
from typing import Final

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")

SMALL_DATASET_LEN: Final[int] = 10
MIN_SAMPLES_PER_CLASS: Final[int] = 2


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for XGBoost experiments with unique logger per experiment"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create unique logger name based on log directory path
    logger_name = f"xgboost_experiment_{hash(str(log_dir))}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # File handler
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train_xgboost_model(X_train, y_train, X_val, y_val, logger, from_config=False, **xgb_params):
    """Train XGBoost model"""

    # Convert to numpy array if it's a pandas Series or DataFrame
    if hasattr(y_train, "values"):
        y_train = y_train.to_numpy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    # Handle edge cases
    if n_pos == 0:
        logger.warning("No positive samples found, setting scale_pos_weight to 1")
        neg_pos = 1.0
    elif n_neg == 0:
        logger.warning("No negative samples found, setting scale_pos_weight to 1")
        neg_pos = 1.0
    else:
        neg_pos = n_neg / n_pos

    if from_config:
        # Default parameters
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 10,
        }
        default_params["scale_pos_weight"] = neg_pos
        default_params.update(xgb_params)
    else:
        default_params = {
            "scale_pos_weight": neg_pos,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 24,
            "use_label_encoder": False,
            "n_jobs": -1,
            "verbosity": 0,
        }

    logger.info(f"XGBoost parameters: {default_params}")
    model = xgb.XGBClassifier(**default_params)

    # Only use eval_set if validation data is different from training data
    # Check if validation data is actually different (not just same reference)
    if len(X_val) != len(X_train) or not np.array_equal(
        X_val.to_numpy() if hasattr(X_val, "values") else X_val,
        X_train.to_numpy() if hasattr(X_train, "values") else X_train,
    ):
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        # Train without validation set (for small datasets)
        model.fit(X_train, y_train, verbose=False)
    return model


def grid_search_xgboost(X_train, y_train, logger, cfg=None):
    """Perform GridSearch to find best hyperparameters"""

    if cfg is None:
        raise ArgumentError("No config was provided!")

    # Define default parameter grid for GridSearch
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    # Base parameters
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    logger.info("Starting GridSearch for hyperparameter tuning...")
    logger.info(f"Parameter grid: {param_grid}")

    # Create base model
    base_model = xgb.XGBClassifier(**base_params)

    # Get CV folds from config or use default
    cv_folds = cfg.xgboost.gridsearch.cv_folds

    # Perform GridSearch
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    # Fit GridSearch
    try:
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = (time.time() - start_time) / 60

        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"GridSearch completed in {search_time:.2f} minutes")
        logger.info(f"Best CV score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

    except ValueError as e:
        logger.warning(f"GridSearch failed with ValueError: {e}")
        logger.info("Falling back to training on single split without cross-validation...")

        # Fallback: train on single split with default parameters
        start_time = time.time()
        best_model = xgb.XGBClassifier(**base_params)
        best_model.fit(X_train, y_train)
        search_time = (time.time() - start_time) / 60

        # Use default parameters
        best_params = {}
        # Calculate score on training data as fallback
        y_pred_proba = best_model.predict_proba(X_train)[:, 1]
        best_score = roc_auc_score(y_train, y_pred_proba)

        logger.info(f"Single split training completed in {search_time:.2f} minutes")
        logger.info(f"Training score: {best_score:.4f}")
        logger.info(f"Using default parameters: {base_params}")

        return best_model, best_params, best_score, search_time

    # Create model with best parameters
    best_model = xgb.XGBClassifier(**base_params, **best_params)

    return best_model, best_params, best_score, search_time


def compute_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive classification metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC calculation
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    metrics["pr_auc"] = auc(recall, precision)

    return metrics


def main_loop(cfg: DictConfig, dataset_name: str, base_dir: Path, dataset_path: str | list | Path):  # noqa
    """Main loop for processing a single dataset"""

    # Check if GridSearch is enabled
    use_gridsearch = cfg.xgboost.gridsearch.enabled

    dataset_dir = base_dir / "xgboost"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(dataset_dir)
    logger.info(f"Starting XGBoost experiment: {dataset_name}")
    logger.info(f"GridSearch enabled: {use_gridsearch}")

    try:
        # Load data
        if isinstance(dataset_path, (str, Path)):
            data = pd.read_csv(dataset_path)
            if isinstance(dataset_path, str):
                data_split = pd.read_csv(dataset_path.replace(".node_features.csv", ".graph.csv"))
            else:
                data_split = pd.read_csv(
                    dataset_path.parent
                    / dataset_path.name.replace(".node_features.csv", ".graph.csv")
                )

            train_data = data.loc[~data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
            test_data = data.loc[data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
            X_train = train_data.drop("target", axis=1)
            y_train = train_data["target"].astype(int)
            X_test = test_data.drop("target", axis=1)
            y_test = test_data["target"].astype(int)

            # Validate that target contains only binary values
            unique_train = set(y_train.unique())
            unique_test = set(y_test.unique())
            if not unique_train.issubset({0, 1}) or not unique_test.issubset({0, 1}):
                logger.error(
                    "Target contains non-binary values. "
                    f"Train: {unique_train}, Test: {unique_test}"
                )
                raise ValueError("Target must contain only 0 and 1 values")
        elif isinstance(dataset_path, list):
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for _dataset_name in dataset_path:
                data = pd.read_csv(dataset_name)
                data_split = pd.read_csv(_dataset_name.replace(".node_features.csv", ".graph.csv"))
                train_data = data.loc[
                    ~data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :
                ]
                test_data = data.loc[
                    data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :
                ]
                X_train.append(train_data.drop("target", axis=1))
                y_train.append(train_data["target"].astype(int))
                X_test.append(test_data.drop("target", axis=1))
                y_test.append(test_data["target"].astype(int))
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
            X_test = pd.concat(X_test, ignore_index=True)
            y_test = pd.concat(y_test, ignore_index=True)

            # Validate that target contains only binary values
            # TODO: requires fix  # noqa: TD002
            unique_train = set(y_train.unique())  # type: ignore
            unique_test = set(y_test.unique())  # type: ignore
            if not unique_train.issubset({0, 1}) or not unique_test.issubset({0, 1}):
                logger.error(
                    "Target contains non-binary values. "
                    f"Train: {unique_train}, Test: {unique_test}"
                )
                raise ValueError("Target must contain only 0 and 1 values")

        # Ensure target is integer
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        data_size = len(X_train) + len(X_test)
        logger.info(f"Data size: {data_size}")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(
            f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}"
        )

        # Check for empty datasets
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error(f"Empty dataset: Train={len(X_train)}, Test={len(X_test)}")
            raise ValueError("Empty dataset detected")

        if use_gridsearch:
            # Use GridSearch to find best hyperparameters
            logger.info("Using GridSearch for hyperparameter optimization...")

            model, best_params, best_cv_score, search_time = grid_search_xgboost(
                X_train, y_train, logger, cfg
            )

            # Train final model with best parameters on full training data
            logger.info("Training final model with best parameters on full training data...")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = (time.time() - start_time) / 60 + search_time

            # Log best parameters
            logger.info(f"Best CV score: {best_cv_score:.4f}")
            logger.info(f"Best parameters: {best_params}")

        else:
            # Use default parameters from config
            xgb_params = {
                "max_depth": cfg.xgboost.max_depth,
                "learning_rate": cfg.xgboost.learning_rate,
                "n_estimators": cfg.xgboost.n_estimators,
                "subsample": cfg.xgboost.subsample,
                "colsample_bytree": cfg.xgboost.colsample_bytree,
            }

            # Train model with default parameters
            # Check if we have enough data for validation split
            min_samples_per_class = min(np.bincount(y_train))

            if (
                len(y_train) >= SMALL_DATASET_LEN
                and min_samples_per_class >= MIN_SAMPLES_PER_CLASS
            ):
                # Use validation split only if we have enough data
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=24, stratify=y_train
                )
                X_train = X_train_split
                y_train = y_train_split
                logger.info(
                    f"Using validation split: {len(X_train)} train, {len(X_val)} validation"
                )
            else:
                # For small datasets, use all training data without validation split
                # Don't use test set for validation to avoid data leakage
                X_val = X_train  # Use same data for training and "validation" (no real validation)
                y_val = y_train
                logger.info(
                    f"Small dataset detected ({len(y_train)} samples),"
                    " using all training data without validation split"
                )
            start_time = time.time()
            model = train_xgboost_model(
                X_train, y_train, X_val, y_val, logger, from_config=False, **xgb_params
            )
            train_time = (time.time() - start_time) / 60
            best_params = xgb_params
            best_cv_score = None

        # Predictions on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log results
        logger.info(f"\n{'=' * 50}")
        logger.info(f"FINAL RESULTS: {dataset_name}/XGBoost")
        logger.info(f"Data size: {data_size}")
        if use_gridsearch:
            logger.info(f"Best CV score: {best_cv_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
        else:
            logger.info(f"Max depth: {best_params['max_depth']}")
            logger.info(f"Learning rate: {best_params['learning_rate']}")
            logger.info(f"N estimators: {best_params['n_estimators']}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        logger.info(f"Total time: {train_time:.2f} minutes")
        logger.info("=" * 50)

        # Save results to CSV file
        results_file = dataset_dir / "results.csv"

        # Check if headers need to be created
        file_exists = results_file.exists()

        row = [
            dataset_name,
            "XGBoost_GridSearch" if use_gridsearch else "XGBoost",
            f"{test_metrics['roc_auc']:.4f}",
            f"{test_metrics['f1']:.4f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{test_metrics['pr_auc']:.4f}",
            f"{train_time:.2f}",
            f"{data_size}",
        ]

        with Path(results_file).open("a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write headers if file is new
            if not file_exists:
                headers = [
                    "dataset_name",
                    "model_type",
                    "test_roc_auc",
                    "test_f1",
                    "test_accuracy",
                    "test_pr_auc",
                    "train_time_minutes",
                    "data_size",
                ]
                writer.writerow(headers)
            writer.writerow(row)

        logger.info(f"Results appended to {results_file}")

        # Save model
        model_path = dataset_dir / "xgboost_model.pkl"
        with Path(model_path).open("wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        # Save best parameters if GridSearch was used
        if use_gridsearch:
            params_path = dataset_dir / "best_parameters.json"
            with Path(params_path).open("w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)
            logger.info(f"Best parameters saved to {params_path}")

    except Exception as e:
        error_msg = f"Error processing dataset {dataset_name}: {str(e)}"
        print(error_msg)
        logger.error(f"{error_msg}\n{traceback.format_exc()}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """XGBoost training script for tabular data"""

    # Initialize output directory
    base_dir = Path(cfg.save_path) / "logs" / str(cfg.data.dataset_size)

    # Load datasets
    if cfg.expand_features:
        dataset_paths = list(
            Path(f"{cfg.data.dataset_path}/csv_{cfg.data.dataset_size}/noisy").glob(
                "*.node_features.csv"
            )
        )
    else:
        dataset_paths = list(
            Path(f"{cfg.data.dataset_path}/csv_{cfg.data.dataset_size}").glob(
                "*.node_features.csv"
            )
        )
    dataset_names = [path.stem for path in dataset_paths]

    if cfg.per_dataset:
        # Process each dataset separately
        for i, dataset_name in enumerate(dataset_names):
            cur_dir = base_dir / dataset_name
            main_loop(cfg, dataset_name, cur_dir, dataset_paths[i])
    else:
        cur_dir = base_dir / "combined_datasets"
        # Process combined data
        main_loop(cfg, "combined_datasets", cur_dir, dataset_paths)


if __name__ == "__main__":
    main()
