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


class InsightReporterAgent:
    def __init__(self):
        self.name = "InsightReporterAgent"
        self.output_dir = "data/reports"
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------- Load Helpers --------------------------- #
    def _load_json(self, path):
        """Load a JSON file if it exists."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------- Insight Logic --------------------------- #
    def _generate_insights(self, eda_summary, model_summary):
        """Generate text-based insights using basic rules."""
        insights = []
        insights.append("=== Automated Insights Report ===\n")

        # 1️⃣ Basic Info
        insights.append("**General Overview:**")
        insights.append(f"- EDA includes {len(eda_summary.get('columns', []))} features.")
        insights.append(f"- Model Type: {model_summary.get('model_name', 'Unknown')}")
        insights.append(f"- Task Type: {model_summary.get('task_type', 'Unknown')}")
        insights.append("")

        # 2️⃣ Data Quality
        missing_info = eda_summary.get("missing_values", {})
        if missing_info:
            high_missing = [col for col, pct in missing_info.items() if pct > 30]
            if high_missing:
                insights.append("⚠️ Some features have high missing values:")
                for col in high_missing:
                    insights.append(f"   - {col}: {missing_info[col]}% missing")
            else:
                insights.append("✅ Missing values are minimal across all features.")
        insights.append("")

        # 3️⃣ Outliers and Distribution
        outlier_info = eda_summary.get("outliers", {})
        if outlier_info:
            top_outliers = sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append("📊 Top features with outliers:")
            for feature, count in top_outliers:
                insights.append(f"   - {feature}: {count} outliers detected")
        insights.append("")

        # 4️⃣ Correlations
        corr_pairs = eda_summary.get("top_correlations", [])
        if corr_pairs:
            insights.append("🔗 Strongest Correlations:")
            for pair in corr_pairs[:3]:
                insights.append(f"   - {pair}")
        insights.append("")

        # 5️⃣ Model Performance
        metrics = model_summary.get("metrics", {})
        if model_summary.get("task_type") == "classification":
            acc = metrics.get("accuracy")
            if acc is not None:
                if acc > 0.9:
                    insights.append(f"✅ Model shows excellent accuracy ({acc * 100:.2f}%).")
                elif acc > 0.75:
                    insights.append(f"🟢 Model accuracy is good ({acc * 100:.2f}%).")
                else:
                    insights.append(f"⚠️ Model accuracy is moderate ({acc * 100:.2f}%). Consider improving features or model type.")
        elif model_summary.get("task_type") == "regression":
            r2 = metrics.get("r2")
            mse = metrics.get("mse")
            if r2 is not None:
                insights.append(f"📈 R² Score: {r2:.3f}")
                if r2 > 0.8:
                    insights.append("✅ Strong predictive capability.")
                elif r2 > 0.5:
                    insights.append("🟢 Moderate performance.")
                else:
                    insights.append("⚠️ Weak performance — model may be underfitting.")
            if mse is not None:
                insights.append(f"📉 MSE: {mse:.3f}")
        insights.append("")

        # 6️⃣ Recommendations
        insights.append("💡 Recommendations:")
        insights.append("- Consider feature scaling or normalization for better model stability.")
        insights.append("- Try feature selection or PCA if the dataset has high dimensionality.")
        insights.append("- Evaluate alternative models (e.g., Gradient Boosting, XGBoost) for improved results.")
        insights.append("- Use cross-validation for more robust performance estimates.")

        return "\n".join(insights)

    # --------------------------- Main Process --------------------------- #
    def process(self, input_data):
        """
        input_data: dict
        {
            "eda_path": str,
            "model_path": str
        }
        """
        eda_path = input_data.get("eda_path")
        model_path = input_data.get("model_path")

        if not eda_path or not model_path:
            raise ValueError("Both 'eda_path' and 'model_path' are required.")

        print(f"[{self.name}] Loading summaries...")
        eda_summary = self._load_json(eda_path)
        model_summary = self._load_json(model_path)

        print(f"[{self.name}] Generating rule-based insights...")
        insights_text = self._generate_insights(eda_summary, model_summary)

        # Save text and JSON reports
        report_txt_path = os.path.join(self.output_dir, "insights_report.txt")
        report_json_path = os.path.join(self.output_dir, "insights_report.json")

        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(insights_text)

        report_data = {
            "eda_summary": eda_summary,
            "model_summary": model_summary,
            "generated_insights": insights_text
        }
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print(f"[{self.name}] Report generated successfully.")
        return {
            "insights_path": report_txt_path,
            "report_json": report_json_path
        }


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = InsightReporterAgent()
    test_input = {
        "eda_path": "data/eda_summary.json",
        "model_path": "data/models/sample_model_summary.json"
    }
    result = agent.process(test_input)
    print("\n✅ Insight Report Generated:\n", result)
