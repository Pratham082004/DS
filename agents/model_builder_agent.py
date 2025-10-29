"""
Agent 2: ModelBuilderAgent (LLM-Integrated)
-------------------------------------------
Automatically detects target column, determines task type (classification/regression),
trains a simple model using sklearn, and — if an LLM is provided —
asks it to reason about feature importance, model choice, and interpretation.

Outputs:
- Model details
- Performance metrics
- Optional LLM reasoning
- Saved model path
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
    def __init__(self, llm=None):
        """
        llm: Optional function that accepts a string prompt and returns a string response.
             Example: llm(prompt) -> str
        """
        self.name = "ModelBuilderAgent"
        self.llm = llm
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # --------------------------- Target Detection --------------------------- #
    def _detect_target_column(self, df):
        """Automatically detect the target column."""
        candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if candidates:
            target_col = min(candidates, key=lambda col: df[col].nunique())
            print(f"[ModelBuilderAgent] Auto-detected categorical target: {target_col}")
            return target_col

        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            target_col = min(num_cols, key=lambda col: df[col].nunique())
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

    # --------------------------- Model Training --------------------------- #
    def _train_classification(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if y.dtype.name == "category" or y.dtype == object:
            y = y.astype("category").cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
            "model_path": model_path
        }

    def _train_regression(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
            "model_path": model_path
        }

    # --------------------------- LLM Integration --------------------------- #
    def _generate_llm_reasoning(self, df, target_col, model_result):
        """If LLM is available, generate reasoning about model and data."""
        if not self.llm:
            return None

        prompt = f"""
        You are an expert data scientist.

        Dataset Columns:
        {list(df.columns)}

        Target Column: {target_col}
        Task Type: {model_result['task_type']}
        Model Used: {model_result['model_name']}
        Metrics: {json.dumps(model_result['metrics'], indent=2)}

        Provide a short, structured reasoning explaining:
        - Why this target column likely represents the prediction goal
        - Why this model type fits the data
        - What the performance metrics indicate
        - Key recommendations for improvement
        """
        try:
            return self.llm(prompt)
        except Exception as e:
            print(f"[ModelBuilderAgent] LLM reasoning failed: {e}")
            return None

    # --------------------------- Core Process --------------------------- #
    def process(self, input_data):
        dataset_path = input_data.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required.")

        df = pd.read_csv(dataset_path)

        # Auto-detect target if not provided
        target_col = input_data.get("target") or self._detect_target_column(df)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        task_type = self._detect_task_type(df, target_col)
        result = self._train_classification(df, target_col) if task_type == "classification" else self._train_regression(df, target_col)

        # Add reasoning if LLM available
        reasoning = self._generate_llm_reasoning(df, target_col, result)
        if reasoning:
            result["llm_reasoning"] = reasoning.strip()

        # Save summary JSON
        json_path = os.path.join(self.model_dir, f"{target_col}_model_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"[ModelBuilderAgent] Model training completed: {result['model_name']}")
        return result
