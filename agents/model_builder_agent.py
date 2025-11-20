"""
Agent 2: ModelBuilderAgent (Enhanced & Fixed Version)
-----------------------------------------------------
Automatically detects the target column, determines task type (classification/regression),
trains the best model using sklearn, and exports clean summary output.

Fixes:
- Uses `type_of_target` for reliable task detection
- Prevents 'unknown label type: continuous' error
- Adds fail-safe for all models
- Uses joblib for model saving
- Improves debug logging
"""

import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.multiclass import type_of_target

warnings.filterwarnings("ignore")


class ModelBuilderAgent:
    def __init__(self):
        self.name = "ModelBuilderAgent"
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # --------------------------- Target Detection --------------------------- #
    def _detect_target_column(self, df):
        """Automatically detect the target column based on dtype and uniqueness."""
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if cat_cols:
            target_col = min(cat_cols, key=lambda c: df[c].nunique())
            print(f"[ModelBuilderAgent] Auto-detected categorical target: {target_col}")
            return target_col

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            target_col = min(num_cols, key=lambda c: df[c].nunique())
            print(f"[ModelBuilderAgent] Auto-detected numeric target: {target_col}")
            return target_col

        raise ValueError("❌ No suitable target column found in dataset.")

    # --------------------------- Task Detection --------------------------- #
    def _detect_task_type(self, y):
        """Reliable detection of classification/regression."""
        y_type = type_of_target(y)
        if y_type in ["binary", "multiclass"]:
            return "classification"
        elif y_type in ["continuous", "continuous-multioutput"]:
            return "regression"
        else:
            # Fallback rule
            return "classification" if len(np.unique(y)) <= 10 else "regression"

    # --------------------------- Data Preparation --------------------------- #
    def _prepare_data(self, df, target_col):
        """Encode categorical columns and scale numeric ones."""
        df = df.copy()

        # Encode all categorical/object columns
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        # Ensure y numeric for models
        if y.dtype in ["object", "category", "bool"]:
            y = LabelEncoder().fit_transform(y.astype(str))

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y

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
            "SVC": SVC(probability=True),
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
        """Train multiple models and find the best one."""
        results = {}
        best_model = None
        best_score = -np.inf

        for name, model in models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                elapsed = time.time() - start

                if task_type == "classification":
                    acc = accuracy_score(y_test, preds)
                    results[name] = acc
                    if acc > best_score:
                        best_score = acc
                        best_model = (name, model, round(acc, 4), elapsed)
                else:
                    r2 = r2_score(y_test, preds)
                    results[name] = r2
                    if r2 > best_score:
                        best_score = r2
                        best_model = (name, model, round(r2, 4), elapsed)

            except Exception as e:
                print(f"[ModelBuilderAgent] ⚠️ {name} failed: {e}")

        if not best_model:
            raise RuntimeError("❌ No model could be successfully trained. Check your dataset.")

        return best_model, results

    # --------------------------- Main Process --------------------------- #
    def process(self, input_data):
        dataset_path = input_data.get("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        target_col = input_data.get("target") or self._detect_target_column(df)
        if target_col not in df.columns:
            raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")

        X, y = self._prepare_data(df, target_col)
        task_type = self._detect_task_type(y)

        print(f"[ModelBuilderAgent] 🔍 Detected task type: {task_type}")
        print(f"[ModelBuilderAgent] Target: {target_col} | Shape: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = (
            self._get_classification_models()
            if task_type == "classification"
            else self._get_regression_models()
        )

        best_model, all_scores = self._train_and_evaluate(X_train, X_test, y_train, y_test, models, task_type)
        best_model_name, model, best_score, training_time = best_model

        # Save model
        model_path = os.path.join(self.model_dir, f"{target_col}_{best_model_name}.joblib")
        joblib.dump(model, model_path)

        # Feature importance (if supported)
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(X.columns, model.feature_importances_.round(4).tolist()))

        result = {
            "task_type": task_type,
            "target_column": target_col,
            "best_model": best_model_name,
            "metric_used": "accuracy" if task_type == "classification" else "r2_score",
            "best_score": best_score,
            "training_time_sec": round(training_time, 3),
            "all_model_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "feature_importance": feature_importance,
            "model_path": model_path,
        }

        summary_path = os.path.join(self.model_dir, f"{target_col}_{best_model_name}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(f"\n✅ Best Model: {best_model_name} ({result['metric_used']} = {best_score})")
        print(f"📦 Model saved at: {model_path}")
        print(f"📊 Summary saved at: {summary_path}\n")

        return result


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = ModelBuilderAgent()
    test_input = {"dataset_path": "data/sample_dataset.csv"}
    result = agent.process(test_input)
    print(json.dumps(result, indent=2))
