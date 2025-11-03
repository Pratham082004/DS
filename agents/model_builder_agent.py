"""
Agent 2: ModelBuilderAgent (Enhanced Non-LLM Version)
-----------------------------------------------------
Automatically detects target column, determines task type (classification/regression),
and trains the best model using sklearn.

Outputs:
- Trained model (.pkl)
- Summary JSON (for InsightsAgent)
- Performance metrics
- Feature importances (if applicable)
"""

import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier, RandomForestRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

    # --------------------------- Data Preparation --------------------------- #
    def _prepare_data(self, df, target_col):
        """Encode categorical features and scale numeric data."""
        df = df.copy()

        # Encode categorical features
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == "object" or y.dtype.name == "category":
            y = LabelEncoder().fit_transform(y.astype(str))

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X, y

    # --------------------------- Classification Models --------------------------- #
    def _get_classification_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RidgeClassifier": RidgeClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "BaggingClassifier": BaggingClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "GaussianNB": GaussianNB(),
            "SVC": SVC(),
            "MLPClassifier": MLPClassifier(max_iter=500),
        }

    # --------------------------- Regression Models --------------------------- #
    def _get_regression_models(self):
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "ExtraTreesRegressor": ExtraTreesRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "BaggingRegressor": BaggingRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "MLPRegressor": MLPRegressor(max_iter=500),
        }

    # --------------------------- Training & Evaluation --------------------------- #
    def _train_and_evaluate(self, X_train, X_test, y_train, y_test, models, task_type):
        """Train multiple models and pick the best based on metric."""
        results = {}
        best_model = None
        best_score = -np.inf

        for name, model in models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                end = time.time()

                if task_type == "classification":
                    acc = accuracy_score(y_test, preds)
                    results[name] = acc
                    if acc > best_score:
                        best_score = acc
                        best_model = (name, model, round(acc, 4), end - start)
                else:
                    r2 = r2_score(y_test, preds)
                    results[name] = r2
                    if r2 > best_score:
                        best_score = r2
                        best_model = (name, model, round(r2, 4), end - start)
            except Exception as e:
                print(f"[ModelBuilderAgent] ⚠️ Skipping {name}: {str(e)}")

        return best_model, results

    # --------------------------- Core Process --------------------------- #
    def process(self, input_data):
        dataset_path = input_data.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Detect target
        target_col = input_data.get("target") or self._detect_target_column(df)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        task_type = self._detect_task_type(df, target_col)
        X, y = self._prepare_data(df, target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = (
            self._get_classification_models()
            if task_type == "classification"
            else self._get_regression_models()
        )

        best_model, all_scores = self._train_and_evaluate(
            X_train, X_test, y_train, y_test, models, task_type
        )

        best_model_name, model, best_score, training_time = best_model

        # Save best model
        model_path = os.path.join(self.model_dir, f"{target_col}_{best_model_name}.pkl")
        pd.to_pickle(model, model_path)

        # Feature importances (if supported)
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(
                zip(X.columns, model.feature_importances_.round(4).tolist())
            )

        # Build summary
        result = {
            "task_type": task_type,
            "best_model": best_model_name,
            "metric_used": "accuracy" if task_type == "classification" else "r2_score",
            "best_score": best_score,
            "training_time_sec": round(training_time, 3),
            "all_model_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "feature_importance": feature_importance,
            "model_path": model_path,
        }

        # ✅ Correct summary filename for orchestrator
        summary_path = os.path.join(
            self.model_dir, f"{target_col}_{best_model_name}_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(f"[ModelBuilderAgent] ✅ Best Model: {best_model_name} (score: {best_score})")
        print(f"[ModelBuilderAgent] Summary saved to: {summary_path}")

        return result


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = ModelBuilderAgent()
    test_input = {"dataset_path": "data/sample_dataset.csv"}
    result = agent.process(test_input)
    print("\n✅ Model Training Summary:\n", json.dumps(result, indent=2))
