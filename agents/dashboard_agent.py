"""
agents/dashboard_agent.py

DashboardAgent
--------------
Generates a simple HTML dashboard combining EDA, model summary, explainability,
and insights files produced by the pipeline.

Usage:
    agent = DashboardAgent()
    result = agent.process({
        "eda_path": "data/eda_summary.json",
        "model_summary_path": "data/models/target_model_summary.json",
        "explainability_path": "data/reports/target_explainability_summary.json",
        "insights_path": "data/reports/insights_report.json"
    })
"""

import os
import json
import time
from typing import Dict, Any, Optional


class DashboardAgent:
    def __init__(self, output_dir: str = "data/dashboard"):
        self.name = "DashboardAgent"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------- Utilities ----------
    def _log(self, msg: str):
        print(f"[DashboardAgent] {msg}", flush=True)

    def _safe_load_json(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {}
        if not os.path.exists(path):
            self._log(f"Warning: file not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self._log(f"Warning: failed to parse JSON {path}: {e}")
            return {}

    def _fmt_percent(self, v):
        try:
            return f"{float(v)*100:.2f}%" if 0 <= float(v) <= 1 else f"{float(v):.2f}"
        except Exception:
            return str(v)

    # ---------- HTML Helpers ----------
    def _html_head(self, title="Auto Dashboard"):
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: Inter, Roboto, Arial, sans-serif; margin: 20px; color: #111; }}
    .card {{ border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 16px; margin-bottom: 16px; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 8px; }}
    h2 {{ font-size: 1.1rem; margin: 6px 0; }}
    pre {{ background:#f7f7f8; padding:12px; border-radius:8px; overflow:auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align:left; padding:8px; border-bottom:1px solid #eee; }}
    .muted {{ color:#666; font-size:0.95rem; }}
  </style>
</head>
<body>
"""

    def _html_footer(self):
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return f"""
  <footer class="muted">Generated at {now} by DashboardAgent</footer>
</body>
</html>
"""

    # ---------- Main process ----------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        input_data keys:
          - eda_path
          - model_summary_path
          - explainability_path (optional)
          - insights_path (optional)
        """
        start = time.time()
        eda_path = input_data.get("eda_path")
        model_summary_path = input_data.get("model_summary_path")
        explain_path = input_data.get("explainability_path")
        insights_path = input_data.get("insights_path")

        self._log("Loading source files...")
        eda = self._safe_load_json(eda_path)
        model = self._safe_load_json(model_summary_path)
        explain = self._safe_load_json(explain_path)
        insights = self._safe_load_json(insights_path)

        title = model.get("best_model", "Auto Model") if isinstance(model, dict) else "Auto Dashboard"
        filename = os.path.join(self.output_dir, "dashboard.html")
        self._log(f"Rendering dashboard to {filename} ...")

        # Compose sections
        sections = []

        # Overview
        total_features = len(eda.get("columns", [])) if isinstance(eda, dict) else None
        task_type = model.get("task_type") if isinstance(model, dict) else None
        best_model = model.get("best_model") if isinstance(model, dict) else None
        best_score = model.get("best_score") if isinstance(model, dict) else None

        overview_html = '<div class="card"><h1>Overview</h1>'
        overview_html += f"<p class='muted'>Task type: <strong>{task_type or 'Unknown'}</strong></p>"
        overview_html += f"<p class='muted'>Best model: <strong>{best_model or 'N/A'}</strong></p>"
        if best_score is not None:
            overview_html += f"<p class='muted'>Best score: <strong>{self._fmt_percent(best_score)}</strong></p>"
        if total_features is not None:
            overview_html += f"<p class='muted'>Total features analyzed: <strong>{total_features}</strong></p>"
        overview_html += "</div>"
        sections.append(overview_html)

        # EDA summary
        eda_html = '<div class="card"><h2>EDA Summary</h2>'
        if eda:
            # show top-level EDA keys and some highlights
            missing = eda.get("missing_percent") or eda.get("missing_values") or {}
            if isinstance(missing, dict) and missing:
                eda_html += "<h3>Missing Values (top 10)</h3><table><thead><tr><th>Column</th><th>Missing</th></tr></thead><tbody>"
                for k, v in list(sorted(missing.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0, reverse=True) )[:10]:
                    eda_html += f"<tr><td>{k}</td><td>{self._fmt_percent(v)}</td></tr>"
                eda_html += "</tbody></table>"
            else:
                eda_html += "<p class='muted'>No missing value summary available.</p>"

            corrs = eda.get("top_correlations") or eda.get("strong_correlations") or []
            if corrs:
                eda_html += "<h3>Top Correlations</h3><pre>" + "\n".join([str(c) for c in corrs[:10]]) + "</pre>"
        else:
            eda_html += "<p class='muted'>No EDA summary found.</p>"
        eda_html += "</div>"
        sections.append(eda_html)

        # Model summary
        model_html = '<div class="card"><h2>Model Summary</h2>'
        if model:
            model_html += "<table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>"
            for k in ("task_type", "target_column", "best_model", "metric_used", "best_score", "training_time_sec"):
                if k in model:
                    model_html += f"<tr><td>{k}</td><td>{self._safe_str(model.get(k))}</td></tr>"
            # show small table of other model scores if present
            scores = model.get("all_model_scores") or model.get("model_scores")
            if isinstance(scores, dict) and scores:
                model_html += "</tbody></table>"
                model_html += "<h3>All model scores</h3><table><thead><tr><th>Model</th><th>Score</th></tr></thead><tbody>"
                for name, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    model_html += f"<tr><td>{name}</td><td>{self._fmt_percent(sc)}</td></tr>"
                model_html += "</tbody></table>"
            else:
                model_html += "</tbody></table>"
        else:
            model_html += "<p class='muted'>No model summary found.</p>"
        model_html += "</div>"
        sections.append(model_html)

        # Explainability
        explain_html = '<div class="card"><h2>Explainability / Feature Importance</h2>'
        if explain and isinstance(explain, dict):
            fi = explain.get("feature_importance") or model.get("feature_importance")
            if isinstance(fi, dict) and fi:
                explain_html += "<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>"
                for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20]:
                    explain_html += f"<tr><td>{feat}</td><td>{self._fmt_percent(imp)}</td></tr>"
                explain_html += "</tbody></table>"
            else:
                # maybe explain file contains textual summary
                text = explain.get("explain_text") or explain.get("summary") or ""
                if text:
                    explain_html += "<pre>" + str(text)[:4000] + ("..." if len(str(text)) > 4000 else "") + "</pre>"
                else:
                    explain_html += "<p class='muted'>No explainability details available.</p>"
        else:
            explain_html += "<p class='muted'>No explainability summary found.</p>"
        explain_html += "</div>"
        sections.append(explain_html)

        # Insights
        insights_html = '<div class="card"><h2>Automated Insights</h2>'
        if insights and isinstance(insights, dict):
            # insights may be in JSON or a text file
            itext = insights.get("insights_text") or insights.get("insights") or ""
            if itext:
                insights_html += "<pre>" + str(itext)[:4000] + ("..." if len(str(itext)) > 4000 else "") + "</pre>"
            else:
                insights_html += "<p class='muted'>No insights text found inside JSON.</p>"
        else:
            # sometimes insights_path points to a txt file; try to load raw
            if insights_path and os.path.exists(insights_path):
                try:
                    with open(insights_path, "r", encoding="utf-8") as f:
                        raw = f.read(4000)
                        insights_html += "<pre>" + raw + ( "..." if len(raw) == 4000 else "") + "</pre>"
                except Exception as e:
                    insights_html += f"<p class='muted'>Failed to read insights file: {e}</p>"
            else:
                insights_html += "<p class='muted'>No insights file found.</p>"
        insights_html += "</div>"
        sections.append(insights_html)

        # Assemble HTML
        html = self._html_head(title=f"Dashboard — {best_model or 'Model'}")
        for s in sections:
            html += s
        html += self._html_footer()

        # Write to disk
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

        duration = round(time.time() - start, 3)
        self._log(f"Dashboard written: {filename} (took {duration}s)")
        return {"dashboard_path": filename, "duration_sec": duration}

    # helper safe string
    def _safe_str(self, v):
        try:
            if v is None:
                return "None"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        except Exception:
            return repr(v)


# ---------- Local test ----------
if __name__ == "__main__":
    agent = DashboardAgent()
    sample_input = {
        "eda_path": "data/eda_summary.json",
        "model_summary_path": "data/models/sample_summary.json",
        "explainability_path": "data/reports/sample_explainability_summary.json",
        "insights_path": "data/reports/insights_report.json"
    }
    print(agent.process(sample_input))
