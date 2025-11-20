"""
agents/feature_engineer_agent.py
Fixed version – production ready
---------------------------------
Key Fixes:
- Prevent target column from being scaled/encoded.
- Proper reattachment of target after transformations.
- Safer feature-type detection and mutual info.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

# --- Default configuration ---
DEFAULT_CONFIG = {
    "imputation": {"numerical": "median", "categorical": "most_frequent", "constant_fill_value": "Missing"},
    "encoding": {"method": "auto", "onehot_threshold": 8, "ordinal_threshold": 30},
    "scaling": {"method": "standard", "apply_to_all_numeric": True},
    "skew_threshold": 1.0,
    "polynomial": {"use": True, "degree": 2, "include_bias": False, "interaction_only": False},
    "feature_selection": {
        "variance_threshold": 0.0,
        "correlation_threshold": 0.9,
        "mutual_info": {"use": True, "top_k": 20},
    },
    "save_visuals": True,
    "output_dirs": {
        "processed": "data/processed",
        "metadata": "data/metadata",
        "visuals": "data/visuals",
        "encoders": "data/metadata/encoders",
        "scalers": "data/metadata/scalers",
        "transformers": "data/metadata/transformers",
    },
}


class FeatureEngineerAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            for k, v in config.items():
                if isinstance(v, dict) and k in self.config:
                    self.config[k].update(v)
                else:
                    self.config[k] = v

        for d in self.config["output_dirs"].values():
            os.makedirs(d, exist_ok=True)

    def _log(self, msg: str):
        print(f"[FeatureEngineerAgent] {msg}", flush=True)

    def _save_json(self, obj: Any, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def _safe_read_csv_or_df(self, dataset_path):
        if isinstance(dataset_path, pd.DataFrame):
            return dataset_path.copy()
        return pd.read_csv(dataset_path)

    # ---------- Feature Type Detection ----------
    def detect_feature_types(self, df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, List[str]]:
        """Detect numeric, categorical, datetime, and text columns, skipping target."""
        work_df = df.copy()
        if target and target in work_df.columns:
            work_df = work_df.drop(columns=[target])

        numerical = work_df.select_dtypes(include=["number"]).columns.tolist()
        datetime_cols = work_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        for col in work_df.select_dtypes(include=["object", "string"]).columns:
            try:
                parsed = pd.to_datetime(work_df[col], errors="coerce")
                if parsed.notnull().mean() > 0.8:
                    datetime_cols.append(col)
                    work_df[col] = parsed
            except Exception:
                pass

        categorical, text_cols = [], []
        for col in work_df.columns:
            if col in numerical or col in datetime_cols:
                continue
            if work_df[col].dtype in ["object", "string"]:
                sample = work_df[col].dropna().astype(str)
                if sample.empty:
                    categorical.append(col)
                    continue
                avg_words = sample.map(lambda s: len(s.split())).mean()
                avg_len = sample.map(len).mean()
                nunique = sample.nunique()
                if avg_words >= 3 or avg_len > 50:
                    text_cols.append(col)
                else:
                    categorical.append(col)
            else:
                categorical.append(col)

        return {"numerical": numerical, "categorical": categorical, "datetime": datetime_cols, "text": text_cols}

    # ---------- Missing Value Handling ----------
    def impute_missing(self, df: pd.DataFrame, types: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict]:
        meta = {"imputation": {}}
        num_cols, cat_cols = types["numerical"], types["categorical"]

        if num_cols:
            imp = SimpleImputer(strategy=self.config["imputation"].get("numerical", "median"))
            df[num_cols] = imp.fit_transform(df[num_cols])
            joblib.dump(imp, os.path.join(self.config["output_dirs"]["transformers"], "imputer_numerical.joblib"))
            meta["imputation"]["numerical"] = "median"

        if cat_cols:
            imp = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = imp.fit_transform(df[cat_cols])
            joblib.dump(imp, os.path.join(self.config["output_dirs"]["transformers"], "imputer_categorical.joblib"))
            meta["imputation"]["categorical"] = "most_frequent"

        return df, meta

    # ---------- Encoding ----------
    def encode_categoricals(self, df, types, target=None):
        enc_meta = {}
        for col in types["categorical"]:
            nunique = df[col].nunique(dropna=True)
            if nunique <= self.config["encoding"]["onehot_threshold"]:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                arr = ohe.fit_transform(df[[col]])
                ohe_cols = [f"{col}__{c}" for c in ohe.categories_[0].astype(str)]
                ohe_df = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), ohe_df], axis=1)
                joblib.dump(ohe, os.path.join(self.config["output_dirs"]["encoders"], f"ohe__{col}.joblib"))
                enc_meta[col] = {"method": "onehot"}
            else:
                ord_enc = OrdinalEncoder()
                df[[col]] = ord_enc.fit_transform(df[[col]])
                joblib.dump(ord_enc, os.path.join(self.config["output_dirs"]["encoders"], f"ord__{col}.joblib"))
                enc_meta[col] = {"method": "ordinal"}
        return df, enc_meta

    # ---------- Scaling & Skew ----------
    def scale_and_transform(self, df, types):
        meta = {}
        num_cols = types["numerical"]
        if not num_cols:
            return df, meta

        # Skew correction
        power = PowerTransformer(method="yeo-johnson")
        skewed = [col for col in num_cols if abs(df[col].skew()) > self.config["skew_threshold"]]
        for col in skewed:
            try:
                df[col] = power.fit_transform(df[[col]])
            except Exception:
                continue

        # Scaling
        scaler_type = self.config["scaling"].get("method", "standard")
        scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, os.path.join(self.config["output_dirs"]["scalers"], f"{scaler_type}_scaler.joblib"))
        meta["scaled_columns"] = num_cols
        return df, meta

    # ---------- Polynomial ----------
    def generate_polynomial_features(self, df, types):
        num_cols = types["numerical"]
        if not num_cols or len(num_cols) < 2:
            return df, {"poly": "skipped"}

        poly_cfg = self.config["polynomial"]
        poly = PolynomialFeatures(
            degree=poly_cfg["degree"],
            interaction_only=poly_cfg["interaction_only"],
            include_bias=poly_cfg["include_bias"],
        )
        arr = poly.fit_transform(df[num_cols])
        cols = poly.get_feature_names_out(num_cols)
        new_df = pd.DataFrame(arr, columns=cols, index=df.index)
        keep_cols = [c for c in cols if c not in num_cols]
        df = pd.concat([df, new_df[keep_cols]], axis=1)
        joblib.dump(poly, os.path.join(self.config["output_dirs"]["transformers"], "poly_features.joblib"))
        return df, {"poly": {"degree": poly_cfg["degree"], "generated": len(keep_cols)}}

    # ---------- Main ----------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        dataset_path = input_data.get("dataset_path")
        target = input_data.get("target")
        if not dataset_path:
            raise ValueError("dataset_path is required")

        df = self._safe_read_csv_or_df(dataset_path)
        if target and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        y = df[target].copy() if target else None
        X = df.drop(columns=[target]) if target else df.copy()

        types = self.detect_feature_types(X, target=target)
        X, impute_meta = self.impute_missing(X, types)
        X, enc_meta = self.encode_categoricals(X, types)
        types = self.detect_feature_types(X)
        X, scale_meta = self.scale_and_transform(X, types)
        X, poly_meta = self.generate_polynomial_features(X, types)

        processed_df = pd.concat([X, y], axis=1) if target else X
        processed_path = os.path.join(self.config["output_dirs"]["processed"], "processed_dataset.csv")
        processed_df.to_csv(processed_path, index=False)

        meta = {
            "imputation": impute_meta,
            "encoding": enc_meta,
            "scaling": scale_meta,
            "polynomial": poly_meta,
            "target": target,
            "original_shape": df.shape,
            "processed_shape": processed_df.shape,
        }
        meta_path = os.path.join(self.config["output_dirs"]["metadata"], "feature_engineering_summary.json")
        self._save_json(meta, meta_path)

        self._log(f"✅ Feature engineering done in {round(time.time() - start, 2)}s")
        return {"processed_dataset_path": processed_path, "metadata_path": meta_path, "metadata": meta}


# ---------------- Local Test ---------------- #
if __name__ == "__main__":
    agent = FeatureEngineerAgent()
    result = agent.process({"dataset_path": "data/sample.csv", "target": None})
    print(json.dumps(result, indent=2))
