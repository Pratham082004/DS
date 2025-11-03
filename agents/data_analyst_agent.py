"""
Agent 1: DataAnalystAgent (Enhanced EDA Version without LLM)
------------------------------------------------------------
Performs Exploratory Data Analysis (EDA):
- Dataset overview (shape, data types, missing values)
- Summary statistics
- Correlation matrix (for numeric features)
- Visualizations (histograms, boxplots, heatmap)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, json


class DataAnalystAgent:
    def __init__(self):
        self.name = "DataAnalystAgent"
        self.output_dir = "data/plots"
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------- Data Handling --------------------------- #
    def _load_data(self, dataset_path):
        """Load CSV or Excel file into a pandas DataFrame."""
        if isinstance(dataset_path, pd.DataFrame):
            return dataset_path.copy()

        path = str(dataset_path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith((".xls", ".xlsx")):
            return pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")

    # --------------------------- Plot Generation --------------------------- #
    def _generate_plots(self, df):
        """Generate EDA plots and save them as PNGs."""
        plot_paths = {}

        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            return {}

        # Histogram and Boxplot for each numeric column
        for col in num_cols:
            clean_data = df[col].dropna()

            # Histogram
            plt.figure(figsize=(5, 4))
            sns.histplot(clean_data, kde=True, color="#3498db")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            hist_path = os.path.join(self.output_dir, f"hist_{col}.png")
            plt.savefig(hist_path, bbox_inches="tight")
            plt.close()
            plot_paths[f"hist_{col}"] = hist_path

            # Boxplot
            plt.figure(figsize=(5, 3))
            sns.boxplot(x=clean_data, color="#e74c3c")
            plt.title(f"Boxplot of {col}")
            box_path = os.path.join(self.output_dir, f"box_{col}.png")
            plt.savefig(box_path, bbox_inches="tight")
            plt.close()
            plot_paths[f"box_{col}"] = box_path

        # Correlation heatmap
        if len(num_cols) > 1:
            plt.figure(figsize=(6, 5))
            corr = df[num_cols].corr().round(2)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path, bbox_inches="tight")
            plt.close()
            plot_paths["correlation_heatmap"] = heatmap_path

        return plot_paths

    # --------------------------- Core Logic --------------------------- #
    def process(self, dataset_path):
        """Perform EDA and return structured summary."""
        df = self._load_data(dataset_path)

        summary = {
            "overview": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
            },
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }

        num_cols = df.select_dtypes(include="number").columns
        summary["correlation_matrix"] = (
            df[num_cols].corr().round(3).to_dict() if len(num_cols) > 1 else {}
        )

        summary["plots"] = self._generate_plots(df)

        # -------------------- Save JSON Summary -------------------- #
        json_path = "data/eda_summary.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
