"""
Agent 1: DataAnalystAgent (Advanced Non-LLM EDA)
------------------------------------------------
Performs comprehensive Exploratory Data Analysis (EDA):
- Dataset overview, dtypes, and corrections
- Missing value analysis + imputation suggestions
- Outlier detection (IQR + Z-Score)
- Multicollinearity check (VIF)
- Time-based feature detection and trend summary
- Categorical imbalance detection
- Visualization exports (histogram, boxplot, heatmap)
- HTML summary report generation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from datetime import datetime


class DataAnalystAgent:
    def __init__(self):
        self.name = "DataAnalystAgent"
        self.output_dir = "data/plots"
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------- Load Data --------------------------- #
    def _load_data(self, dataset_path):
        """Load dataset (CSV/Excel) into DataFrame."""
        if isinstance(dataset_path, pd.DataFrame):
            return dataset_path.copy()
        if str(dataset_path).endswith(".csv"):
            return pd.read_csv(dataset_path)
        elif str(dataset_path).endswith((".xls", ".xlsx")):
            return pd.read_excel(dataset_path)
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")

    # --------------------------- Auto Type Correction --------------------------- #
    def _auto_type_correction(self, df):
        """Attempt to correct data types automatically."""
        corrections = {}
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                    corrections[col] = "datetime"
                    continue
                except:
                    pass
                try:
                    df[col] = pd.to_numeric(df[col])
                    corrections[col] = "numeric"
                    continue
                except:
                    pass
                if df[col].str.lower().isin(["yes", "no", "true", "false"]).any():
                    df[col] = df[col].map({"yes": 1, "no": 0, "true": 1, "false": 0})
                    corrections[col] = "boolean"
        return df, corrections

    # --------------------------- Missing Value Analysis --------------------------- #
    def _missing_value_report(self, df):
        """Return missing values count, percent, and imputation suggestions."""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df) * 100).round(2)
        suggestions = {}
        for col in df.columns[df.isnull().any()]:
            if df[col].dtype == "object":
                suggestions[col] = "fill with mode"
            elif np.issubdtype(df[col].dtype, np.number):
                suggestions[col] = "fill with mean/median"
            elif np.issubdtype(df[col].dtype, np.datetime64):
                suggestions[col] = "forward/backward fill"
        return missing.to_dict(), missing_percent.to_dict(), suggestions

    # --------------------------- Outlier Detection --------------------------- #
    def _outlier_detection(self, df):
        """Detect outliers using IQR and Z-score methods."""
        numeric_cols = df.select_dtypes(include="number").columns
        outlier_summary = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            iqr_outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
            z_outliers = (np.abs(stats.zscore(series)) > 3).sum()
            outlier_summary[col] = {
                "IQR_outliers": int(iqr_outliers),
                "Z_outliers": int(z_outliers),
            }
        return outlier_summary

    # --------------------------- Multicollinearity (VIF) --------------------------- #
    def _multicollinearity_check(self, df):
        """Detect multicollinearity using Variance Inflation Factor (VIF)."""
        num_df = df.select_dtypes(include="number").dropna()
        if num_df.shape[1] < 2:
            return {}
        vif_data = pd.DataFrame()
        vif_data["feature"] = num_df.columns
        vif_data["VIF"] = [
            round(variance_inflation_factor(num_df.values, i), 3)
            for i in range(num_df.shape[1])
        ]
        return dict(zip(vif_data["feature"], vif_data["VIF"]))

    # --------------------------- Time Feature Detection --------------------------- #
    def _time_analysis(self, df):
        """Detect datetime columns and provide trend overview."""
        datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns
        trends = {}
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            trends[col] = {
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
                "unique_dates": int(df[col].nunique()),
                "frequency": df[col].diff().value_counts().head(3).to_dict(),
            }
        return trends

    # --------------------------- Plot Generation --------------------------- #
    def _generate_plots(self, df):
        """Generate and save EDA plots."""
        plot_paths = {}
        num_cols = df.select_dtypes(include="number").columns

        for col in num_cols:
            clean_data = df[col].dropna()

            plt.figure(figsize=(5, 4))
            sns.histplot(clean_data, kde=True, color="#3498db")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            hist_path = os.path.join(self.output_dir, f"hist_{col}.png")
            plt.savefig(hist_path, bbox_inches="tight")
            plt.close()
            plot_paths[f"hist_{col}"] = hist_path

            plt.figure(figsize=(5, 3))
            sns.boxplot(x=clean_data, color="#e74c3c")
            plt.title(f"Boxplot of {col}")
            box_path = os.path.join(self.output_dir, f"box_{col}.png")
            plt.savefig(box_path, bbox_inches="tight")
            plt.close()
            plot_paths[f"box_{col}"] = box_path

        if len(num_cols) > 1:
            plt.figure(figsize=(6, 5))
            corr = df[num_cols].corr().round(2)
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path, bbox_inches="tight")
            plt.close()
            plot_paths["correlation_heatmap"] = heatmap_path

        return plot_paths

    # --------------------------- Rule-Based Summary --------------------------- #
    def _generate_text_summary(self, df):
        """Generate natural-language summary based on key stats."""
        lines = []
        lines.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        num_cols = len(df.select_dtypes(include='number').columns)
        cat_cols = len(df.select_dtypes(exclude='number').columns)
        lines.append(f"Numeric columns: {num_cols}, Categorical columns: {cat_cols}.")
        missing_ratio = df.isnull().mean().mean() * 100
        if missing_ratio > 10:
            lines.append("Significant missing data detected (>10%).")
        dup = df.duplicated().sum()
        if dup > 0:
            lines.append(f"{dup} duplicate rows detected.")
        return " ".join(lines)

    # --------------------------- Core Process --------------------------- #
    def process(self, dataset_path):
        """Perform enhanced EDA and return structured JSON summary."""
        df = self._load_data(dataset_path)
        df, corrections = self._auto_type_correction(df)

        missing, missing_pct, impute_suggestions = self._missing_value_report(df)
        outlier_summary = self._outlier_detection(df)
        vif_report = self._multicollinearity_check(df)
        time_trends = self._time_analysis(df)
        plot_paths = self._generate_plots(df)
        summary_text = self._generate_text_summary(df)

        summary = {
            "overview": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
            },
            "dtype_corrections": corrections,
            "missing_values": missing,
            "missing_percent": missing_pct,
            "imputation_suggestions": impute_suggestions,
            "outliers": outlier_summary,
            "multicollinearity_vif": vif_report,
            "time_analysis": time_trends,
            "correlation_matrix": df.select_dtypes(include="number").corr().round(3).to_dict(),
            "plots": plot_paths,
            "text_summary": summary_text,
        }

        # Save structured JSON and HTML
        os.makedirs("data", exist_ok=True)
        json_path = "data/eda_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        html_path = "data/eda_summary.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<h2>EDA Summary Report</h2><p>{summary_text}</p>")
            for name, path in plot_paths.items():
                f.write(f"<h4>{name}</h4><img src='{path}' width='500'><br>")

        print(f"[DataAnalystAgent] ✅ EDA summary saved to {json_path} and {html_path}")
        return summary


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = DataAnalystAgent()
    summary = agent.process("data/sample.csv")
    print(json.dumps(summary, indent=2))
