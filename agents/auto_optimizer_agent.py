"""
agents/auto_optimizer_agent.py

AutoOptimizerAgent (Production-ready)
------------------------------------
Purpose:
- Run automated hyperparameter optimization (Optuna, RandomizedSearchCV, GridSearchCV).
- Supports regression and classification.
- Outputs best model, optimization summary, and history plot.
"""

import os
import json
import time
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ---- Optional Optuna ----
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False


DEFAULT_CONFIG = {
    "cv": 5,
    "n_iter": 30,
    "n_trials": 40,
    "search_method": "optuna",
    "n_jobs": -1,
    "random_state": 42,
    "output_dir": "data/models",
    "save_plots": True
}


class AutoOptimizerAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _log(self, msg: str):
        print(f"[AutoOptimizerAgent] {msg}", flush=True)

    def _save_json(self, data: Any, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _infer_task_and_scoring(self, task_type: Optional[str], scoring: Optional[str]) -> Tuple[str, str]:
        if not task_type:
            task_type = "classification"
        if scoring:
            return task_type, scoring
        return (task_type, "accuracy" if task_type == "classification" else "r2")

    def _default_candidates(self, task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            return {
                "LogisticRegression": {
                    "model": LogisticRegression(max_iter=1000, random_state=self.config["random_state"]),
                    "param_dist": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"]}
                },
                "RandomForestClassifier": {
                    "model": RandomForestClassifier(random_state=self.config["random_state"]),
                    "param_dist": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]}
                },
                "KNeighborsClassifier": {
                    "model": KNeighborsClassifier(),
                    "param_dist": {"n_neighbors": [3, 5, 7, 9]}
                }
            }
        else:
            return {
                "Ridge": {"model": Ridge(), "param_dist": {"alpha": [0.01, 0.1, 1.0, 10.0]}},
                "RandomForestRegressor": {
                    "model": RandomForestRegressor(random_state=self.config["random_state"]),
                    "param_dist": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]}
                },
                "SVR": {"model": SVR(), "param_dist": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}}
            }

    # ---- Optuna objective builder ----
    def _optuna_objective_factory(self, X, y, base_model, param_dist, scoring, cv):
        def objective(trial):
            params = {}
            for p, vals in param_dist.items():
                if isinstance(vals, list):
                    params[p] = trial.suggest_categorical(p, vals)
                elif isinstance(vals, dict):
                    low, high = vals.get("low"), vals.get("high")
                    if vals.get("log", False):
                        params[p] = trial.suggest_loguniform(p, low, high)
                    else:
                        params[p] = trial.suggest_uniform(p, low, high)
            model = base_model.__class__(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return float(np.mean(scores))
        return objective

    def _run_optuna(self, X, y, model_name, candidate, scoring, cv):
        param_dist = candidate["param_dist"]
        base_model = candidate["model"]
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.config["random_state"]),
                                    pruner=MedianPruner())
        objective = self._optuna_objective_factory(X, y, base_model, param_dist, scoring, cv)
        study.optimize(objective, n_trials=self.config["n_trials"])
        best_params = study.best_params
        model = base_model.__class__(**best_params)
        model.fit(X, y)
        history = [{"trial": t.number, "value": t.value, "params": t.params} for t in study.trials]
        return model, best_params, study.best_value, history

    def _run_random_search(self, X, y, candidate, scoring, cv):
        rs = RandomizedSearchCV(
            candidate["model"],
            candidate["param_dist"],
            n_iter=self.config["n_iter"],
            scoring=scoring,
            cv=cv,
            n_jobs=self.config["n_jobs"],
            random_state=self.config["random_state"]
        )
        rs.fit(X, y)
        return rs.best_estimator_, rs.best_params_, float(rs.best_score_), rs.cv_results_

    def _run_grid_search(self, X, y, candidate, scoring, cv):
        gs = GridSearchCV(
            candidate["model"],
            candidate["param_dist"],
            scoring=scoring,
            cv=cv,
            n_jobs=self.config["n_jobs"]
        )
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_, float(gs.best_score_), gs.cv_results_

    # ---- Main process ----
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()

        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a dictionary")

        processed_path = (
            input_data.get("processed_dataset_path")
            or input_data.get("processed_path")
            or input_data.get("data_path")
        )
        target = input_data.get("target")

        # Auto-fallback if orchestrator didn't send path
        if not processed_path and os.path.exists("data/processed/processed_dataset.csv"):
            processed_path = "data/processed/processed_dataset.csv"

        if not processed_path or not target:
            raise ValueError("processed_dataset_path and target are required in input_data")

        if isinstance(processed_path, str):
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f"Processed dataset not found: {processed_path}")
            df = pd.read_csv(processed_path)
        else:
            df = processed_path

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        task_type, scoring = self._infer_task_and_scoring(
            input_data.get("task_type"), input_data.get("scoring")
        )
        candidates = input_data.get("models") or self._default_candidates(task_type)
        cv = input_data.get("cv", self.config["cv"])
        search_method = input_data.get("search_method", self.config["search_method"])

        X, y = df.drop(columns=[target]), df[target]
        results, best_overall = {}, {"score": -np.inf}

        for name, candidate in candidates.items():
            try:
                self._log(f"Tuning {name} using {search_method.upper()}...")
                if search_method == "optuna" and OPTUNA_AVAILABLE:
                    model, params, score, history = self._run_optuna(X, y, name, candidate, scoring, cv)
                elif search_method == "grid":
                    model, params, score, history = self._run_grid_search(X, y, candidate, scoring, cv)
                else:
                    model, params, score, history = self._run_random_search(X, y, candidate, scoring, cv)

                eval_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                eval_score = float(np.mean(eval_scores))

                model_path = os.path.join(self.config["output_dir"], f"{name}_optimized.pkl")
                joblib.dump(model, model_path)

                results[name] = {
                    "best_params": params,
                    "cv_score": score,
                    "eval_score": eval_score,
                    "model_path": model_path,
                }

                if eval_score > best_overall["score"]:
                    best_overall = {"name": name, "path": model_path, "score": eval_score, "params": params}

                hist_path = os.path.join(self.config["output_dir"], f"{name}_history.json")
                self._save_json(history, hist_path)
                results[name]["history_path"] = hist_path

                self._log(f"✅ {name} done — eval_score={eval_score:.4f}")
            except Exception as e:
                self._log(f"⚠️ {name} failed: {e}")
                results[name] = {"error": str(e)}

        summary = {
            "task_type": task_type,
            "scoring": scoring,
            "results": results,
            "best_overall": best_overall,
            "duration_sec": round(time.time() - start, 3)
        }

        summary_path = os.path.join(self.config["output_dir"], "optimization_summary.json")
        self._save_json(summary, summary_path)
        self._log(f"Summary saved: {summary_path}")

        # --- Optional visualization ---
        if self.config.get("save_plots", True):
            try:
                valid = {k: v["eval_score"] for k, v in results.items() if "eval_score" in v}
                if valid:
                    plt.figure(figsize=(8, 4))
                    plt.bar(valid.keys(), valid.values())
                    plt.title("Model Evaluation Scores")
                    plt.ylabel(scoring)
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plot_path = os.path.join(self.config["output_dir"], "optimization_scores.png")
                    plt.savefig(plot_path, bbox_inches="tight")
                    plt.close()
                    summary["plot_path"] = plot_path
                    self._log(f"Plot saved: {plot_path}")
            except Exception as e:
                self._log(f"Plot generation failed: {e}")

        return {
            "summary_path": summary_path,
            "best_model": best_overall,
            "results": results,
        }
