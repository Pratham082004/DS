"""
Agent 2: ModelBuilderAgent (Non-LLM Version)
--------------------------------------------
Automatically detects target column, determines task type (classification/regression),
and trains a model using sklearn.

Outputs:
- Trained model (.pkl)
- Summary JSON (for InsightsAgent)
- Performance metrics
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")


class ModelBuilderAgent:
    def __init__(self):
        self.name = "ModelBuilderAgent"
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # --------------------------- Target Detection --------------------------- #
    def _detect_target_column(self, df):
        """Automatically detect target column based on dtype and uniqueness."""
        categorical_candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if categorical_candidates:
            target_col = min(categorical_candidates, key=lambda col: df[col].nunique())
            print(f"[ModelBuilderAgent] Auto-detected categorical target: {target_col}")
            return target_col

        numeric_candidates = df.select_dtypes(include="number").columns.tolist()
        if numeric_candidates:
            target_col = min(numeric_candidates, key=lambda col: df[col].nunique())
            print(f"[ModelBuilderAgent] Auto-detected numeric target: {target_col}")
            return target_col

        raise ValueError("No suitable target column found in dataset.")

    # --------------------------- Task Detection --------------------------- #
    def _detect_task_type(self, df, target_col):
        """Detect if task is classification or regression."""
        target_data = df[target_col]
        if target_data.dtype in ["object", "category", "bool"]:
            return "classification"
        if target_data.nunique() <= 10:
            return "classification"
        return "regression"

    # --------------------------- Classification --------------------------- #
    def _train_classification(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype.name == "category" or y.dtype == object:
            y = y.astype("category").cat.codes

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        model_path = os.path.join(self.model_dir, f"{target_col}_classifier.pkl")
        pd.to_pickle(model, model_path)

        return {
            "task_type": "classification",
            "model_name": "RandomForestClassifier",
            "metrics": {"accuracy": round(acc, 4)},
            "model_path": model_path,
        }

    # --------------------------- Regression --------------------------- #
    def _train_regression(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        model_path = os.path.join(self.model_dir, f"{target_col}_regressor.pkl")
        pd.to_pickle(model, model_path)

        return {
            "task_type": "regression",
            "model_name": "RandomForestRegressor",
            "metrics": {"mse": round(mse, 4), "r2": round(r2, 4)},
            "model_path": model_path,
        }

    # --------------------------- Core Process --------------------------- #
    def process(self, input_data):
        dataset_path = input_data.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Auto-detect target column
        target_col = input_data.get("target") or self._detect_target_column(df)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        task_type = self._detect_task_type(df, target_col)

        # Train appropriate model
        result = (
            self._train_classification(df, target_col)
            if task_type == "classification"
            else self._train_regression(df, target_col)
        )

        # Save summary JSON (for InsightsAgent)
        summary_path = result["model_path"].replace(".pkl", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(f"[ModelBuilderAgent] Model training completed: {result['model_name']}")
        print(f"[ModelBuilderAgent] Model summary saved to: {summary_path}")

        return result


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = ModelBuilderAgent()
    test_input = {"dataset_path": "data/sample_dataset.csv"}
    result = agent.process(test_input)
    print("\n✅ Model Training Summary:\n", json.dumps(result, indent=2))
