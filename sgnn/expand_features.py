import argparse
import glob
import os
import pathlib

import numpy as np
import pandas as pd
from obtain_data import Synolitic
from tqdm import tqdm


def main(args):
    pathlib.Path(os.path.join(args.data_path, "csv_1.0", "noisy")).mkdir(
        exist_ok=True, parents=True
    )
    # Load data with frac = 1.0
    dataset_paths = glob.glob(f"{args.data_path}/csv_1.0/*.node_features.csv")
    for dataset_path in tqdm(dataset_paths, desc="Expanding dimensions"):
        df = pd.read_csv(dataset_path).iloc[:, 1:]
        # Record the list of original columns
        original_cols = [c for c in df.columns if c != "target"]
        for i, col in enumerate(original_cols):
            if col != "target":
                noisy_col = f"feature_{i + len(original_cols)}"
                df[noisy_col] = df[col] * (
                    1
                    + np.random.default_rng(42).uniform(
                        -args.noise_level, args.noise_level, size=len(df)
                    )
                )
        # Get the numeric columns (all columns except the target)
        numeric_cols = [col for col in df.columns if col != "target"]
        # Obtain test ids
        graph_df = pd.read_csv(dataset_path.replace(".node_features.csv", ".graph.csv"))
        test_ids = np.array(list(graph_df.columns[2:]))[graph_df.iloc[-1, 2:].values == 1].astype(
            int
        )
        test_df = df.loc[test_ids, :]
        ytest = test_df["target"]
        Xtest = test_df.drop(columns=["target"])
        # Obtain train ids
        train_ids = np.array(list(graph_df.columns[2:]))[graph_df.iloc[-1, 2:].values == 0].astype(
            int
        )
        train_df = df.loc[train_ids, :]
        ytrain = train_df["target"]
        Xtrain = train_df.drop(columns=["target"])
        # Update features_df and target
        features_df = pd.concat([Xtrain, Xtest])
        target = pd.concat([ytrain, ytest])

        # Create a Synolitic object
        gr = Synolitic(
            classifier_str="svc",
            probability=True,
            random_state=37,
            numeric_cols=numeric_cols,
            category_cols=None,
        )

        # Fit the Synolitic model on the training data
        gr.fit(X_train=Xtrain, y_train=ytrain)

        # Predict the labels for the test data
        _ = gr.predict(X_test=features_df)

        # Add train/test indicator row
        train_col = ["is_in_test", "-"]
        for i in gr.graph_df.columns[2:]:
            if int(i) not in list(Xtrain.index):
                train_col.append(1)
            else:
                train_col.append(0)

        gr.graph_df.loc[gr.graph_df.shape[0]] = train_col

        # Save the results to a CSV file
        path_to_save = os.path.join(
            args.data_path,
            "csv_1.0",
            "noisy",
            dataset_path.split("/")[-1].split(".")[0] + ".graph" + ".csv",
        )
        gr.graph_df.to_csv(path_to_save, index=False)

        df_path_to_save = os.path.join(
            args.data_path,
            "csv_1.0",
            "noisy",
            dataset_path.split("/")[-1].split(".")[0] + ".node_features" + ".csv",
        )
        # Add target back to features_df before saving
        features_df_with_target = features_df.copy()
        features_df_with_target["target"] = target
        features_df_with_target.to_csv(df_path_to_save, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--noise_level", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
