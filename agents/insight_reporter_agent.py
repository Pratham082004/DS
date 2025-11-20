"""
InsightReporterAgent (Safe + Robust)
-----------------------------------
Now prevents accidental loading of .joblib / .pkl as JSON.
More robust handling of model summary structures.
"""

import os
import json
import time
import numpy as np
from datetime import datetime


class InsightReporterAgent:
    def __init__(self):
        self.name = "InsightReporterAgent"
        self.output_dir = "data/reports"
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # ✅ FIX #1 — SAFE JSON LOADER
    # ---------------------------------------------------------------------- #
    def _load_json(self, path):
        """Load a JSON file safely, blocking binary files."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Block binary model files
        if path.endswith(".joblib") or path.endswith(".pkl"):
            raise ValueError(
                f"Refusing to load binary model file as JSON: {path}\n"
                "Pass *_summary.json instead."
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing JSON at {path}: {e}")

    # ---------------------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------------------- #
    def _safe_value(self, val):
        if isinstance(val, np.generic):
            return val.item()
        return val if isinstance(val, (dict, list, str, int, float, bool)) else str(val)

    # ---------------------------------------------------------------------- #
    # ✅ FIX #2 — Robust model summary extractor
    # ---------------------------------------------------------------------- #
    def _extract_best_model_info(self, model_summary):
        """
        Handles multiple possible model summary formats:
        - ModelBuilderAgent
        - AutoOptimizerAgent
        - Custom structures
        """
        # Try AutoOptimizer format
        if "best_overall" in model_summary:
            best = model_summary["best_overall"]
            return (
                best.get("name") or best.get("model") or "Unknown",
                best.get("score") or None
            )

        # Try ModelBuilderAgent format
        if "best_model" in model_summary:
            return (
                model_summary.get("best_model"),
                model_summary.get("best_score")
            )

        if "model_name" in model_summary:
            return (
                model_summary.get("model_name"),
                model_summary.get("metrics", {}).get("accuracy")
                or model_summary.get("metrics", {}).get("r2")
            )

        # Fallback
        return "Unknown", None

    # ---------------------------------------------------------------------- #
    # Generate Insights
    # ---------------------------------------------------------------------- #
    def _generate_insights(self, eda_summary, model_summary):
        print("[InsightReporterAgent] 🔍 Generating structured insights...", flush=True)
        insights = ["=== 📊 Automated Insights Report ===\n"]

        # Extract model info safely
        model_name, best_score = self._extract_best_model_info(model_summary)
        task_type = model_summary.get("task_type", "Unknown")

        # -------- General Overview -------- #
        try:
            all_columns = (
                list(eda_summary.get("columns", [])) or
                list(eda_summary.get("dtype_corrections", {}).keys())
            )

            insights.append("**General Overview:**")
            insights.append(f"- Total Features: {len(all_columns)}")
            insights.append(f"- Model Used: {self._safe_value(model_name)}")
            insights.append(f"- Task Type: {self._safe_value(task_type)}")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Overview: {e}\n")

        # -------- Missing Values -------- #
        try:
            missing = (
                eda_summary.get("missing_percent", {})
                or eda_summary.get("missing_values", {})
                or {}
            )

            if missing:
                high_missing = [
                    col for col, pct in missing.items()
                    if pct and float(pct) > 30
                ]
                if high_missing:
                    insights.append("⚠️ High Missing Value Columns:")
                    for col in high_missing:
                        insights.append(f"  - {col}: {missing[col]}%")
                else:
                    insights.append("✅ Minimal missing values.")
            else:
                insights.append("✅ No missing values.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Missing Value Error: {e}\n")

        # -------- Outliers -------- #
        try:
            outliers = eda_summary.get("outliers", {})
            if outliers:
                top5 = sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:5]
                insights.append("📊 Top Outlier Features:")
                for f, c in top5:
                    insights.append(f"  - {f}: {c} outliers")
            else:
                insights.append("✅ No major outliers.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Outlier Error: {e}\n")

        # -------- Correlations -------- #
        try:
            corrs = eda_summary.get("top_correlations", [])
            if corrs:
                insights.append("🔗 Strongest Correlations:")
                for pair in corrs[:5]:
                    insights.append(f"  - {pair}")
            else:
                insights.append("ℹ️ No strong correlations found.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Correlation Error: {e}\n")

        # -------- Model Performance -------- #
        try:
            insights.append("**Model Performance:**")
            metrics = model_summary.get("metrics", {})

            if task_type == "classification":
                acc = best_score or metrics.get("accuracy")
                if acc is not None:
                    acc = float(acc)
                    insights.append(f"Accuracy: {acc:.3f}")
                    if acc > 0.9: insights.append("✅ Excellent classification performance.")
                    elif acc > 0.75: insights.append("🟢 Good performance.")
                    else: insights.append("⚠️ Consider tuning or more data.")

            elif task_type == "regression":
                r2 = best_score or metrics.get("r2")
                if r2 is not None:
                    r2 = float(r2)
                    insights.append(f"R² Score: {r2:.3f}")
                if metrics.get("mse") is not None:
                    insights.append(f"MSE: {float(metrics['mse']):.3f}")

            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Model Metrics Error: {e}\n")

        # -------- Feature Importance -------- #
        try:
            fi = model_summary.get("feature_importance")
            if fi:
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
                insights.append("🔥 Top Influential Features:")
                for k, v in top:
                    insights.append(f"  - {k}: {v}")
                insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Feature Importance Error: {e}\n")

        # -------- Recommendations -------- #
        insights.append("💡 **Recommendations:**")
        insights.extend([
            "- Try hyperparameter tuning for improvement.",
            "- Feature selection can boost performance.",
            "- Consider boosting models like XGBoost/LightGBM.",
            "- Evaluate using cross-validation.",
            "- Normalize/standardize features if needed."
        ])

        return "\n".join(insights)

    # ---------------------------------------------------------------------- #
    # Main Process
    # ---------------------------------------------------------------------- #
    def process(self, input_data):
        start = time.time()
        eda_path = input_data.get("eda_path")
        model_path = input_data.get("model_path")

        if not eda_path or not model_path:
            raise ValueError("Both 'eda_path' and 'model_path' must be provided.")

        print(f"[{self.name}] Loading JSON summaries...", flush=True)
        eda_summary = self._load_json(eda_path)
        model_summary = self._load_json(model_path)

        insights_text = self._generate_insights(eda_summary, model_summary)

        report_txt = os.path.join(self.output_dir, "insights_report.txt")
        report_json = os.path.join(self.output_dir, "insights_report.json")

        result_data = {
            "generated_at": datetime.now().isoformat(),
            "duration_sec": round(time.time() - start, 2),
            "eda_summary_keys": list(eda_summary.keys()),
            "model_summary_keys": list(model_summary.keys()),
            "insights_text": insights_text,
        }

        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(insights_text)
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)

        print(f"[{self.name}] ✅ Insight report generated.")
        return {
            "insights_path": report_txt,
            "report_json": report_json
        }
