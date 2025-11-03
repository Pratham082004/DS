"""
Agent 3: InsightReporterAgent (No LLM)
--------------------------------------
Combines EDA + Model summaries and generates rule-based insights.

Outputs:
- insights_report.txt
- insights_report.json
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

    # --------------------------- Safe JSON Loader --------------------------- #
    def _load_json(self, path):
        """Load a JSON file safely."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                raise ValueError(f"Error parsing JSON at {path}: {e}")

    # --------------------------- Utility --------------------------- #
    def _safe_value(self, val):
        """Safely convert NumPy / invalid types to native Python."""
        if isinstance(val, (np.generic,)):
            return val.item()
        if isinstance(val, (dict, list, str, int, float, bool)) or val is None:
            return val
        return str(val)

    # --------------------------- Core Insight Generator --------------------------- #
    def _generate_insights(self, eda_summary, model_summary):
        print("[InsightReporterAgent] 🔍 Step 1: Starting insight generation...", flush=True)
        insights = ["=== 📊 Automated Insights Report ===\n"]

        try:
            # 1️⃣ General Overview
            print("[InsightReporterAgent] Step 2: General overview...", flush=True)
            insights.append("**General Overview:**")
            all_columns = list(eda_summary.get("columns", [])) or list(eda_summary.get("dtype_corrections", {}).keys())
            insights.append(f"- Total Features Analyzed: {len(all_columns)}")
            insights.append(f"- Model Name: {self._safe_value(model_summary.get('model_name', 'Unknown'))}")
            insights.append(f"- Task Type: {self._safe_value(model_summary.get('task_type', 'Unknown'))}")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in General Overview: {e}")

        try:
            # 2️⃣ Missing Values
            print("[InsightReporterAgent] Step 3: Data quality (missing values)...", flush=True)
            missing_info = eda_summary.get("missing_percent", {}) or eda_summary.get("missing_values", {})
            if missing_info:
                high_missing = [col for col, pct in missing_info.items() if self._safe_value(pct) and pct > 30]
                if high_missing:
                    insights.append("⚠️ High Missing Value Columns:")
                    for col in high_missing:
                        insights.append(f"   - {col}: {missing_info[col]}% missing")
                else:
                    insights.append("✅ Minimal missing values across all features.")
            else:
                insights.append("✅ No missing values detected.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Missing Values: {e}")

        try:
            # 3️⃣ Outlier Analysis
            print("[InsightReporterAgent] Step 4: Outlier analysis...", flush=True)
            outlier_info = eda_summary.get("outliers", {})
            if outlier_info:
                sorted_outliers = sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)[:5]
                insights.append("📊 Top Features with Outliers:")
                for feature, count in sorted_outliers:
                    insights.append(f"   - {feature}: {count} outliers detected")
            else:
                insights.append("✅ No major outliers detected.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Outlier Analysis: {e}")

        try:
            # 4️⃣ Correlation Insights
            print("[InsightReporterAgent] Step 5: Correlation analysis...", flush=True)
            top_corrs = eda_summary.get("top_correlations", [])
            if top_corrs:
                insights.append("🔗 Strongest Correlations:")
                for pair in top_corrs[:5]:
                    insights.append(f"   - {pair}")
            else:
                insights.append("ℹ️ No strong correlations found.")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Correlation Insights: {e}")

        try:
            # 5️⃣ Model Performance
            print("[InsightReporterAgent] Step 6: Model performance...", flush=True)
            metrics = model_summary.get("metrics", {})
            task_type = model_summary.get("task_type", "unknown")

            if task_type == "classification":
                acc = self._safe_value(metrics.get("accuracy"))
                if acc is not None:
                    if acc >= 0.9:
                        insights.append(f"✅ Model shows excellent accuracy ({acc * 100:.2f}%).")
                    elif acc >= 0.75:
                        insights.append(f"🟢 Model accuracy is good ({acc * 100:.2f}%).")
                    else:
                        insights.append(f"⚠️ Model accuracy is moderate ({acc * 100:.2f}%). Consider feature engineering or hyperparameter tuning.")
            elif task_type == "regression":
                r2 = self._safe_value(metrics.get("r2"))
                mse = self._safe_value(metrics.get("mse"))
                if r2 is not None:
                    insights.append(f"📈 R² Score: {r2:.3f}")
                    if r2 > 0.8:
                        insights.append("✅ Strong predictive capability.")
                    elif r2 > 0.5:
                        insights.append("🟢 Moderate performance.")
                    else:
                        insights.append("⚠️ Weak predictive performance. Try more features or regularization.")
                if mse is not None:
                    insights.append(f"📉 Mean Squared Error (MSE): {mse:.3f}")
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Model Performance: {e}")

        try:
            # 6️⃣ Recommendations
            print("[InsightReporterAgent] Step 7: Adding recommendations...", flush=True)
            insights.append("💡 **Recommendations:**")
            insights.extend([
                "- Consider feature scaling or normalization for better model stability.",
                "- Try dimensionality reduction (PCA) if dataset has many features.",
                "- Evaluate advanced models (e.g., XGBoost, LightGBM, CatBoost).",
                "- Apply cross-validation for more reliable performance.",
                "- Tune hyperparameters using GridSearchCV or Optuna.",
                "- Check feature importance to remove low-impact columns."
            ])
            insights.append("")
        except Exception as e:
            insights.append(f"⚠️ Error in Recommendations: {e}")

        print("[InsightReporterAgent] ✅ All sections processed successfully.", flush=True)
        return "\n".join(insights)

    # --------------------------- Main Process --------------------------- #
    def process(self, input_data):
        """
        input_data = {
            "eda_path": "path/to/eda_summary.json",
            "model_path": "path/to/model_summary.json"
        }
        """
        start_time = time.time()

        eda_path = input_data.get("eda_path")
        model_path = input_data.get("model_path")

        if not eda_path or not model_path:
            raise ValueError("Both 'eda_path' and 'model_path' must be provided.")

        print(f"[{self.name}] Loading EDA & Model summaries...", flush=True)
        eda_summary = self._load_json(eda_path)
        model_summary = self._load_json(model_path)

        print(f"[{self.name}] Generating rule-based insights...", flush=True)
        insights_text = self._generate_insights(eda_summary, model_summary)

        report_txt_path = os.path.join(self.output_dir, "insights_report.txt")
        report_json_path = os.path.join(self.output_dir, "insights_report.json")

        print(f"[{self.name}] Writing reports to {self.output_dir}...", flush=True)
        try:
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write(insights_text)

            report_data = {
                "generated_at": datetime.now().isoformat(),
                "duration_sec": round(time.time() - start_time, 2),
                "eda_summary_keys": list(eda_summary.keys()),
                "model_summary_keys": list(model_summary.keys()),
                "insights_text": insights_text
            }

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to write reports: {e}")

        print(f"[{self.name}] ✅ Report generated successfully in {round(time.time() - start_time, 2)}s", flush=True)
        return {
            "insights_path": report_txt_path,
            "report_json": report_json_path
        }


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = InsightReporterAgent()
    test_input = {
        "eda_path": "data/eda_summary.json",
        "model_path": "data/models/Species_LogisticRegression_summary.json"
    }
    result = agent.process(test_input)
    print("\n✅ Insight Report Generated:\n", result)
