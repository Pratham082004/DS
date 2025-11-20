"""
agents/model_explainer_agent.py

Enhanced ModelExplainerAgent
----------------------------
Now automatically detects:
- Most recent model file from data/models/
- Associated target and task type from JSON summary (if available)
- Most recent dataset from data/processed/
"""

import os
import json
import time
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance, PartialDependenceDisplay, partial_dependence
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings("ignore")

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False


DEFAULT_CONFIG = {
    "output_dir": "data/reports",
    "n_top_features": 8,
    "n_local_explanations": 3,
    "pdp_features": 3,
    "shap_sample_size": 200,
    "permutation_n_repeats": 10,
    "random_state": 42,
    "save_plots": True,
}


class ModelExplainerAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _log(self, msg: str):
        print(f"[ModelExplainerAgent] {msg}", flush=True)

    def _save_json(self, obj: Any, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    # ---------- Loaders ---------- #
    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._log(f"Loading model from {model_path}")
        return joblib.load(model_path)

    def _load_dataset(self, dataset_path: str):
        if isinstance(dataset_path, pd.DataFrame):
            return dataset_path.copy()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        self._log(f"Loading dataset from {dataset_path}")
        return pd.read_csv(dataset_path)

    # ---------- Feature Importances ---------- #
    def _model_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        fi = {}
        try:
            if hasattr(model, "feature_importances_"):
                vals = np.array(model.feature_importances_)
                fi = dict(zip(feature_names, vals.round(6)))
            elif hasattr(model, "coef_"):
                vals = np.abs(np.array(model.coef_).ravel())
                fi = dict(zip(feature_names, vals.round(6)))
        except Exception as e:
            self._log(f"Model-based importance failed: {e}")
        return fi

    def _permutation_importance(self, model, X, y, scoring=None):
        try:
            self._log("Computing permutation importances...")
            res = permutation_importance(
                model, X, y,
                n_repeats=self.config["permutation_n_repeats"],
                random_state=self.config["random_state"],
                n_jobs=-1,
                scoring=scoring
            )
            imp = dict(zip(X.columns.tolist(), res.importances_mean.round(6)))
            return imp, res
        except Exception as e:
            self._log(f"Permutation importance failed: {e}")
            return {}, None

    # ---------- SHAP ---------- #
    def _compute_shap(self, model, X, task_type: Optional[str]):
        if not SHAP_AVAILABLE:
            self._log("SHAP not installed.")
            return None
        try:
            X_sample = X.sample(min(self.config["shap_sample_size"], len(X)),
                                random_state=self.config["random_state"])
            if hasattr(model, "feature_importances_"):
                expl = shap.TreeExplainer(model)
                shap_values = expl.shap_values(X_sample)
            else:
                expl = shap.Explainer(model, X_sample)
                shap_values = expl(X_sample)
            return {"explainer": expl, "shap_values": shap_values, "X_sample": X_sample}
        except Exception as e:
            self._log(f"SHAP computation failed: {e}")
            return None

    # ---------- PDP ---------- #
    def _save_pdp(self, model, X, features: List[str], model_name: str):
        saved = []
        for feat in features[: self.config["pdp_features"]]:
            if feat not in X.columns:
                continue
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                try:
                    PartialDependenceDisplay.from_estimator(model, X, [feat], ax=ax)
                except Exception:
                    pdp = partial_dependence(model, X, [feat])
                    xs = pdp["values"][0]
                    ys = pdp["average"][0]
                    ax.plot(xs, ys)
                ax.set_title(f"PDP: {feat}")
                path = os.path.join(self.config["output_dir"], f"{model_name}_pdp_{feat}.png")
                plt.tight_layout()
                plt.savefig(path, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
            except Exception as e:
                self._log(f"PDP failed for {feat}: {e}")
        return saved

    # ---------- Local explanations ---------- #
    def _local_explain(self, model, X, indices: List[int], model_name: str):
        saved = []
        if SHAP_AVAILABLE:
            try:
                shap_obj = self._compute_shap(model, X, None)
                if shap_obj:
                    shap_vals = shap_obj["shap_values"]
                    X_sample = shap_obj["X_sample"]
                    for idx in indices[: self.config["n_local_explanations"]]:
                        fig = plt.figure(figsize=(6, 3))
                        try:
                            if isinstance(shap_vals, list):
                                vals = shap_vals[0]
                            else:
                                vals = shap_vals
                            shap.plots.waterfall(
                                shap.Explanation(values=vals[idx],
                                                 base_values=0,
                                                 data=X_sample.iloc[idx]),
                                show=False
                            )
                        except Exception:
                            shap.summary_plot(shap_vals, X_sample, show=False)
                        path = os.path.join(self.config["output_dir"], f"{model_name}_local_{idx}.png")
                        plt.tight_layout()
                        plt.savefig(path, bbox_inches="tight")
                        plt.close(fig)
                        saved.append(path)
            except Exception as e:
                self._log(f"Local SHAP failed: {e}")

        if not saved and LIME_AVAILABLE:
            try:
                explainer = LimeTabularExplainer(
                    training_data=X.values,
                    feature_names=X.columns.tolist(),
                    mode='classification' if hasattr(model, "predict_proba") else 'regression'
                )
                for idx in indices[: self.config["n_local_explanations"]]:
                    exp = explainer.explain_instance(
                        X.iloc[idx].values,
                        model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                        num_features=min(10, X.shape[1])
                    )
                    fig = exp.as_pyplot_figure()
                    path = os.path.join(self.config["output_dir"], f"{model_name}_local_lime_{idx}.png")
                    fig.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(path)
            except Exception as e:
                self._log(f"LIME failed: {e}")
        return saved

    # ---------- Main process ---------- #
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        model_path = input_data.get("model_path")
        dataset_path = input_data.get("dataset_path")
        target = input_data.get("target")
        task_type = input_data.get("task_type")
        local_indices = input_data.get("local_indices", [])

        if not model_path or not dataset_path:
            raise ValueError("Both model_path and dataset_path required.")

        model = self._load_model(model_path)
        df = self._load_dataset(dataset_path)

        # ---- Ensure dataset columns match model ---- #
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [c for c in expected if c not in df.columns]
            extra = [c for c in df.columns if c not in expected]
            if missing:
                self._log(f"⚠️ Missing expected columns: {missing}")
            if extra:
                self._log(f"⚠️ Dropping extra columns not seen during training: {extra}")
                df = df.drop(columns=extra, errors="ignore")
            X = df[expected] if all(c in df.columns for c in expected) else df
        else:
            if target and target in df.columns:
                df = df.drop(columns=[target])
            if target:
                df = df.loc[:, ~df.columns.str.contains(target, case=False, regex=False)]
            X = df.copy()

        y = None
        if target and target in df.columns:
            y = df[target]

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        summary = {
            "model_name": model_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance": {},
            "model_based_importance": {},
            "permutation_importance": {},
            "top_features": [],
            "pdp_plots": [],
            "local_explanations": [],
            "shap": {"available": SHAP_AVAILABLE, "summary_plot": None},
        }

        # ---- Performance ---- #
        try:
            if y is not None and len(y) == len(X):
                preds = model.predict(X)
                if task_type == "regression":
                    summary["performance"]["r2"] = round(r2_score(y, preds), 4)
                else:
                    summary["performance"]["accuracy"] = round(accuracy_score(y, preds), 4)
        except Exception as e:
            self._log(f"Metric computation failed: {e}")

        # ---- Importances ---- #
        mfi = self._model_feature_importance(model, X.columns.tolist())
        summary["model_based_importance"] = dict(sorted(mfi.items(), key=lambda x: x[1], reverse=True))

        pi, _ = self._permutation_importance(
            model, X, y,
            scoring=("r2" if task_type == "regression" else "accuracy") if y is not None else None
        )
        summary["permutation_importance"] = dict(sorted(pi.items(), key=lambda x: x[1], reverse=True))

        # ---- Top features ---- #
        top_features = list(dict.fromkeys(
            list(summary["model_based_importance"].keys()) +
            list(summary["permutation_importance"].keys())
        ))[: self.config["n_top_features"]]
        summary["top_features"] = top_features

        # ---- SHAP Summary ---- #
        if SHAP_AVAILABLE:
            try:
                shap_obj = self._compute_shap(model, X, task_type)
                if shap_obj:
                    shap_values = shap_obj["shap_values"]
                    X_sample = shap_obj["X_sample"]
                    fig = plt.figure(figsize=(8, 6))
                    shap.summary_plot(
                        shap_values[0] if isinstance(shap_values, list) else shap_values,
                        X_sample,
                        show=False
                    )
                    shap_plot_path = os.path.join(self.config["output_dir"], f"{model_name}_shap_summary.png")
                    plt.tight_layout()
                    plt.savefig(shap_plot_path, bbox_inches="tight")
                    plt.close(fig)
                    summary["shap"]["summary_plot"] = shap_plot_path
            except Exception as e:
                self._log(f"SHAP summary failed: {e}")

        # ---- Feature Importance Plot ---- #
        try:
            imp_dict = summary["model_based_importance"] or summary["permutation_importance"]
            if imp_dict:
                names, vals = zip(*list(imp_dict.items())[: self.config["n_top_features"]])
                fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
                ax.barh(range(len(names))[::-1], vals[::-1])
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names[::-1])
                ax.set_xlabel("Importance")
                ax.set_title("Top Feature Importances")
                path = os.path.join(self.config["output_dir"], f"{model_name}_feature_importance.png")
                plt.tight_layout()
                plt.savefig(path, bbox_inches="tight")
                plt.close(fig)
                summary["feature_importance_plot"] = path
        except Exception as e:
            self._log(f"Feature importance plot failed: {e}")

        # ---- PDP & Local ---- #
        summary["pdp_plots"] = self._save_pdp(model, X, top_features, model_name)
        chosen = local_indices or list(range(min(self.config["n_local_explanations"], len(X))))
        summary["local_explanations"] = self._local_explain(model, X, chosen, model_name)

        # ---- Save Summary ---- #
        summary_path = os.path.join(self.config["output_dir"], f"{model_name}_explainability_summary.json")
        self._save_json(summary, summary_path)
        self._log(f"Explainability summary saved: {summary_path}")

        elapsed = round(time.time() - start, 3)
        self._log(f"Completed in {elapsed}s")

        return {"summary_path": summary_path, "summary": summary}


# ---------- Auto-detect latest model & dataset ---------- #
if __name__ == "__main__":
    agent = ModelExplainerAgent()

    # Find latest model
    model_dir = "data/models"
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if not model_files:
        print("❌ No model files found in data/models.")
        exit(1)
    latest_model = max(model_files, key=os.path.getmtime)

    # Try to find corresponding summary JSON
    summary_path = latest_model.replace(".joblib", "_summary.json")
    target, task_type = None, None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            target = meta.get("target_column")
            task_type = meta.get("task_type")

    # Find latest dataset
    dataset_dir = "data/processed"
    dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    if not dataset_files:
        print("❌ No dataset files found in data/processed.")
        exit(1)
    latest_dataset = max(dataset_files, key=os.path.getmtime)

    print(f"\n[Auto-detect] Using model: {latest_model}")
    print(f"[Auto-detect] Using dataset: {latest_dataset}")
    print(f"[Auto-detect] Target: {target} | Task Type: {task_type}\n")

    try:
        res = agent.process({
            "model_path": latest_model,
            "dataset_path": latest_dataset,
            "target": target,
            "task_type": task_type,
        })
        print(json.dumps(res["summary"], indent=2))
    except Exception as ex:
        print("Error running ModelExplainerAgent:", ex)
