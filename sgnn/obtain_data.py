import argparse
import operator
import os
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def mat_to_dataframe(mat_path, output_dir):
    """Load a .mat file and convert it into a pandas DataFrame with features and target."""
    data = loadmat(mat_path, simplify_cells=True)

    Y = data["Y"]
    X = data["X"]
    if len(Y.shape) > 1 or len(X.shape) < 2:
        return 0

    # Create column names for features
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]

    # Build DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = Y
    df = df.sample(frac=1, random_state=42)

    path_to_save = os.path.join(output_dir, f"{mat_path.split('/')[-1]}.csv")
    df.to_csv(path_to_save)
    return 1


class Synolitic:
    def __init__(
        self,
        classifier_str: str,
        probability: bool = False,
        random_state: int | None = None,
        numeric_cols: list | None = None,
        category_cols: list | None = None,
    ):
        self.classifier_str = classifier_str
        self.probability = probability
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.category_cols = category_cols
        self.nodes_tpl_list: list | None = None
        self.preprocessor: ColumnTransformer | None = None
        self.clf_dict: dict | None = None
        self.predicts: list | None = None
        self.graph_df: pd.DataFrame | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Preprocesses the data, fits classifiers, and creates a graph.

        Parameters:
        X_train (pd.DataFrame): The input features.
        y_train (pd.Series): The target variable.

        Returns:
        None
        """

        # Preprocess numeric and category data
        transformers_list = []

        if self.numeric_cols is not None:
            transformers_list.append(("num", StandardScaler(), self.numeric_cols))
        if self.category_cols is not None:
            transformers_list.append(("cat", OneHotEncoder(), self.category_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers_list, remainder="passthrough"
        )
        self.preprocessor.fit(X_train)

        X_train_processed = pd.DataFrame(
            columns=self.preprocessor.get_feature_names_out(),
            data=self.preprocessor.transform(X_train),
        )

        # Create pairs of features for all features
        self.nodes_tpl_list = list(combinations(iterable=X_train_processed.columns, r=2))

        # Reset classifier list for each fit call
        self.clf_dict = {}

        def aux_fit_clf(
            idx: int,
            df: pd.DataFrame,
            feature_name_1: str,
            feature_name_2: str,
            y_train: pd.Series,
        ) -> tuple:
            """
            Fits a classifier to a pair of features.

            Parameters:
            idx (int): The index of the pair of features.
            df (pd.DataFrame): The processed features.
            feature_name_1 (str): The name of the first feature in the pair.
            feature_name_2 (str): The name of the second feature in the pair.
            y_train (pd.Series): The target variable.

            Returns:
            tuple: A tuple containing the index, feature names, and the fitted classifier.
            """
            clf = (
                SVC(
                    probability=self.probability,
                    class_weight="balanced",
                    random_state=self.random_state,
                )
                if self.classifier_str == "svc"
                else LogisticRegression(class_weight="balanced", random_state=self.random_state)
                if self.classifier_str == "logreg"
                else _raise(exception_type=ValueError, msg="Unknown classifier")
            )
            clf.fit(df[[feature_name_1, feature_name_2]], y_train)
            return idx, feature_name_1, feature_name_2, clf

        # Fill tpl_list on all CPU kernels
        tpl_list = Parallel(n_jobs=-1, verbose=0, prefer="processes")(
            delayed(aux_fit_clf)(
                idx=idx,
                df=X_train_processed,
                feature_name_1=feature_1,
                feature_name_2=feature_2,
                y_train=y_train,
            )
            for idx, (feature_1, feature_2) in enumerate(self.nodes_tpl_list)
        )

        self.graph_df = pd.DataFrame(columns=["p1", "p2"], data=self.nodes_tpl_list)
        self.clf_dict = {
            idx: [feature_1, feature_2, clf] for idx, feature_1, feature_2, clf in tpl_list
        }

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the output for the given test data.

        Parameters:
            X_test (pd.DataFrame): The test data to make predictions on.

        Returns:
            pd.DataFrame: The predicted output for the test data.
        """
        # Process the test data using the preprocessor
        X_test_processed = pd.DataFrame(
            data=self.preprocessor.transform(X_test),  # type: ignore
            index=X_test.index,
            columns=self.preprocessor.get_feature_names_out(),  # type: ignore
        )  # type: ignore

        def aux_predict_clf(X_test: pd.DataFrame, clf: ClassifierMixin) -> tuple:
            """
            Auxiliary function to make predictions using a classifier.

            Parameters:
                X_test (pd.DataFrame): The processed test data.
                clf (ClassifierMixin): The classifier to make predictions with.

            Returns:
                tuple: The predicted output as a tuple.
            """
            # Make predictions using the classifier
            return (
                tuple(clf.predict_proba(X_test)[:, 1])
                if self.probability
                else tuple(clf.predict(X_test))
            )

        # Make predictions for each classifier in clf_dict and update the graph dataframe
        self.graph_df.loc[:, X_test_processed.index] = [
            aux_predict_clf(X_test=X_test_processed[[val[0], val[1]]], clf=val[2])
            for _, val in self.clf_dict.items()
        ]  # type: ignore

        return self.graph_df


def _raise(exception_type, msg):
    raise exception_type(msg)


def get_df_by_frac(frac: float) -> float:
    if frac == 0.9:
        return 1.0
    if frac == 0.7:
        return 0.9
    if frac == 0.5:
        return 0.7
    if frac == 0.4:
        return 0.5
    if frac == 0.2:
        return 0.4
    if frac == 0.1:
        return 0.2
    if frac == 0.05:
        return 0.1
    if frac == 0.01:
        return 0.05
    raise ValueError(f"Fraction {frac} not supported")


def _build_and_save_graph(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features_df: pd.DataFrame,
    target: pd.Series,
    numeric_cols: list,
    base_name: str,
    output_path: pathlib.Path,
) -> None:
    """Build Synolitic graph and save results to CSV files."""
    # Create a Synolitic object
    gr = Synolitic(
        classifier_str="svc",
        probability=True,
        random_state=37,
        numeric_cols=numeric_cols,
        category_cols=None,
    )

    # Fit the Synolitic model on the training data
    gr.fit(X_train=X_train, y_train=y_train)

    # Predict the labels for all data
    _ = gr.predict(X_test=features_df)

    # Add train/test indicator row
    train_col = ["is_in_test", "-"]
    for i in gr.graph_df.columns[2:]:
        if int(i) not in list(X_train.index):
            train_col.append(1)
        else:
            train_col.append(0)

    gr.graph_df.loc[gr.graph_df.shape[0]] = train_col

    # Save the results to CSV files
    path_to_save = output_path / f"{base_name}.graph.csv"
    gr.graph_df.to_csv(path_to_save, index=False)

    df_path_to_save = output_path / f"{base_name}.node_features.csv"
    # Add target back to features_df before saving
    features_df_with_target = features_df.copy()
    features_df_with_target["target"] = target
    features_df_with_target.to_csv(df_path_to_save, index=True)


def main(args):
    # Convert all .mat files in the directory
    input_dir = args.data_path  # Current directory (where .mat files are)
    output_dir = args.output_dir
    csv_dir = f"{output_dir}/csv"
    pathlib.Path(csv_dir).mkdir(exist_ok=True, parents=True)
    if not pathlib.Path(f"{csv_dir}/Banknote.mat.csv").exists():
        for filename in tqdm(
            os.listdir(input_dir),
            total=len(os.listdir(input_dir)),
            desc="Converting .mat files to .csv",
        ):
            if filename.endswith(".mat"):
                mat_path = os.path.join(input_dir, filename)
                res = mat_to_dataframe(mat_path, csv_dir)
                if res != 1:
                    print(f"Error with {filename}")

    files_with_size = [
        (f, pathlib.Path(os.path.join(csv_dir, f)).stat().st_size) for f in os.listdir(csv_dir)
    ]
    # Sort files by size (increasing order)
    files_with_size.sort(key=operator.itemgetter(1))
    files_with_size = files_with_size[:15]

    pathlib.Path(output_dir + f"/csv_{args.data_size}/").mkdir(exist_ok=True, parents=True)

    # Iterate over sorted files
    for filename, _size in tqdm(
        files_with_size,
        total=len(files_with_size),
        desc=f"Building synolitic graphs for data size {args.data_size}",
    ):
        if args.data_size == 1.0:
            path = f"{output_dir}/csv/{filename}"
            df = pd.read_csv(path).iloc[:, 1:]
            # Get the numeric columns (all columns except the target)
            numeric_cols = [col for col in df.columns if col != "target"]
            # Drop the target column from the DataFrame to get the features
            features_df = df.drop(columns=["target"])
            # Get the target column
            target = df["target"]
            
            # Use StratifiedKFold for cross-validation
            skf = StratifiedKFold(
                n_splits=args.n_folds, shuffle=True, random_state=42
            )
            
            # Process each fold
            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(features_df, target)
            ):
                # Create fold directory
                fold_dir = pathlib.Path(
                    output_dir + f"/csv_{args.data_size}/fold_{fold_idx}"
                )
                fold_dir.mkdir(exist_ok=True, parents=True)
                
                # Split data for this fold
                X_train = features_df.iloc[train_idx].copy()
                X_test = features_df.iloc[test_idx].copy()
                y_train = target.iloc[train_idx].copy()
                y_test = target.iloc[test_idx].copy()
                
                # Combine all data for graph building
                features_df_fold = pd.concat([X_train, X_test])
                target_fold = pd.concat([y_train, y_test])
                
                # Reset indices for proper graph construction
                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                features_df_fold = features_df_fold.reset_index(drop=True)
                target_fold = target_fold.reset_index(drop=True)
                
                # Set indices for train/test separation
                X_train.index = range(len(X_train))
                X_test.index = range(len(X_train), len(X_train) + len(X_test))
                
                # Build and save graph
                base_name = filename.split(".")[0]
                _build_and_save_graph(
                    X_train=X_train,
                    y_train=y_train,
                    features_df=features_df_fold,
                    target=target_fold,
                    numeric_cols=numeric_cols,
                    base_name=base_name,
                    output_path=fold_dir,
                )
        elif args.data_size != 1.0:
            prev_data_size = get_df_by_frac(args.data_size)
            base_name = filename.split(".")[0]
            
            # Process each fold
            for fold_idx in range(args.n_folds):
                prev_path = pathlib.Path(
                    output_dir + f"/csv_{prev_data_size}/fold_{fold_idx}/{base_name}.graph.csv"
                )
                if not prev_path.exists():
                    continue
                
                # Read the graph data from previous size for this fold
                df = pd.read_csv(prev_path)
                selected_ids = np.array(list(df.columns[2:]))[df.iloc[-1, 2:].values == 0].astype(int)
                orig_df = pd.read_csv(f"{output_dir}/csv/{filename}").iloc[:, 1:]
                train_df_orig = orig_df.loc[selected_ids, :]
                
                # Obtain test ids
                test_ids = np.array(list(df.columns[2:]))[df.iloc[-1, 2:].values == 1].astype(int)
                test_df = orig_df.loc[test_ids, :]
                
                # Train slice
                train_df = train_df_orig.groupby("target", group_keys=False).sample(
                    frac=args.data_size / prev_data_size, random_state=42
                )
                
                # If no samples after slice, take one sample with target 0 and one with target 1
                if len(train_df) == 0:
                    samples = []
                    for target_val in [0, 1]:
                        target_samples = train_df_orig[train_df_orig["target"] == target_val]
                        if len(target_samples) > 0:
                            samples.append(target_samples.sample(n=1, random_state=42))
                    train_df = pd.concat(samples)
                else:
                    train_df = train_df.sample(frac=1, random_state=42)
                
                # Check if all samples are of one class, add one sample of the other class
                unique_targets = train_df["target"].unique()
                if len(unique_targets) == 1:
                    missing_target = 1 - unique_targets[0]
                    missing_samples = train_df_orig[train_df_orig["target"] == missing_target]
                    # if len(missing_samples) > 0:
                    train_df = pd.concat([train_df, missing_samples.sample(n=1, random_state=42)])
                    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

                # Update Xtrain and ytrain
                y_train = train_df["target"]
                X_train = train_df.drop(columns=["target"])

                y_test = test_df["target"]
                X_test = test_df.drop(columns=["target"])

                # Update features_df and target
                features_df = pd.concat([X_train, X_test])
                target = pd.concat([y_train, y_test])

                # Get the numeric columns (all columns except the target)
                numeric_cols = [col for col in train_df.columns if col != "target"]

                # Create fold directory for current data size
                fold_dir = pathlib.Path(
                    output_dir + f"/csv_{args.data_size}/fold_{fold_idx}"
                )
                fold_dir.mkdir(exist_ok=True, parents=True)

                # Build and save graph
                _build_and_save_graph(
                    X_train=X_train,
                    y_train=y_train,
                    features_df=features_df,
                    target=target,
                    numeric_cols=numeric_cols,
                    base_name=base_name,
                    output_path=fold_dir,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--data_size", type=float, default=1.0)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()
    main(args)
