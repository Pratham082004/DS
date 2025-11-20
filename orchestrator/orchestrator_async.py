"""
orchestrator/async_orchestrator.py

AsyncOrchestrator (AutoOptimizer Removed) — FIXED
-------------------------------------------------
- Robust model_summary_path detection for both .pkl and .joblib
- Robust processed dataset extraction from FeatureEngineerAgent output
- Retry wrapper respects configured attempts/base delay
- No AutoOptimizerAgent; optional OptimizerAgent only
- Never passes a .joblib to InsightReporter (uses *_summary.json)
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AsyncOrchestrator")

# ---------- Safe dynamic imports ----------
def safe_import(module_path: str, class_name: str):
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        logger.info(f"Imported {class_name} from {module_path}")
        return cls
    except Exception as e:
        logger.warning(f"Could not import {class_name} from {module_path}: {e}")
        return None

# Agents
DataAnalystAgent    = safe_import("agents.data_analyst_agent",      "DataAnalystAgent")
FeatureEngineerAgent= safe_import("agents.feature_engineer_agent",  "FeatureEngineerAgent")
ModelBuilderAgent   = safe_import("agents.model_builder_agent",     "ModelBuilderAgent")
OptimizerAgent      = safe_import("agents.optimizer_agent",         "OptimizerAgent")  # manual optimizer (optional)
ModelExplainerAgent = safe_import("agents.model_explainer_agent",   "ModelExplainerAgent")
InsightReporterAgent= safe_import("agents.insight_reporter_agent",  "InsightReporterAgent")
DashboardAgent      = safe_import("agents.dashboard_agent",         "DashboardAgent")
DeploymentAgent     = safe_import("agents.deployment_agent",        "DeploymentAgent")
MemoryManagerAgent  = safe_import("agents.memory_manager_agent",    "MemoryManagerAgent")

# ---------- Utilities ----------
def ensure_dir_for_file(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj: Any, path: str):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

async def run_blocking(loop, fn, *args, **kwargs):
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

def retry_sync(fn, attempts=3, base_delay=1.0, exceptions=(Exception,)):
    """Retry a synchronous function with exponential backoff."""
    def wrapper(*args, **kwargs):
        delay = base_delay
        last_exc = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except exceptions as e:
                last_exc = e
                logger.warning(f"[retry] attempt {i+1}/{attempts} for {fn.__name__} failed: {e}; retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
        logger.error(f"[retry] All {attempts} attempts failed for {fn.__name__}")
        raise last_exc
    return wrapper

def _extract_processed_path(fe_result: Union[str, Dict[str, Any]], fallback: str) -> str:
    """
    Accept multiple common keys or even a plain string path from FeatureEngineerAgent.
    """
    if isinstance(fe_result, str) and fe_result.strip():
        return fe_result.strip()

    if isinstance(fe_result, dict):
        for k in [
            "processed_dataset_path", "processed_path", "processed_dataset",
            "processed_file", "output_path"
        ]:
            v = fe_result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return fallback

def _detect_model_summary_path(model_binary_path: Optional[str], model_dir: str) -> Optional[str]:
    """
    Derive *_summary.json from a model binary path, handling both .pkl and .joblib.
    Fallback: pick the newest *_summary.json in model_dir if present.
    """
    if model_binary_path:
        p = Path(model_binary_path)
        if p.suffix in [".pkl", ".joblib"]:
            candidate = str(p).replace(p.suffix, "_summary.json")
            if Path(candidate).exists():
                return candidate

    # Fallback: newest *_summary.json in model_dir
    md = Path(model_dir)
    if md.exists():
        summaries = sorted(md.glob("*_summary.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if summaries:
            return str(summaries[0])

    return None

# ---------- Orchestrator ----------
class AsyncOrchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_cfg = {
            "output_dir": "data",
            "eda_path": "data/eda_summary.json",
            "model_dir": "data/models",
            "explain_dir": "data/reports",
            "visuals_dir": "data/visuals",
            "run_steps": [
                "eda",
                "feature_engineer",
                "model_build",
                "optimize",      # Only manual optimizer exists (optional)
                "model_explain",
                "insight_report",
                "dashboard",
                "deploy"
            ],
            "register_memory": True,
            "retry_attempts": 3,
            "retry_base_delay": 1.0
        }

        if config:
            default_cfg.update(config)
        self.cfg = default_cfg

        # Instantiate available agents
        self.eda_agent       = DataAnalystAgent() if DataAnalystAgent else None
        self.fe_agent        = FeatureEngineerAgent() if FeatureEngineerAgent else None
        self.model_agent     = ModelBuilderAgent() if ModelBuilderAgent else None
        self.opt_agent       = OptimizerAgent() if OptimizerAgent else None   # optional
        self.explainer_agent = ModelExplainerAgent() if ModelExplainerAgent else None
        self.report_agent    = InsightReporterAgent() if InsightReporterAgent else None
        self.dashboard_agent = DashboardAgent() if DashboardAgent else None
        self.deployment_agent= DeploymentAgent() if DeploymentAgent else None
        self.memory_agent    = MemoryManagerAgent() if MemoryManagerAgent else None

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    # ---------- Async wrappers (respect retry cfg) ----------
    async def run_eda(self, dataset_path: str):
        if not self.eda_agent:
            logger.warning("EDA agent missing — skipping EDA")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.eda_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, dataset_path)

    async def run_feature_engineer(self, dataset_path: str):
        if not self.fe_agent:
            logger.info("FeatureEngineerAgent missing — skipping")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.fe_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {"dataset_path": dataset_path})

    async def run_model_build(self, dataset_path: str, target: Optional[str]):
        if not self.model_agent:
            logger.warning("ModelBuilderAgent missing — skipping model build")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.model_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {"dataset_path": dataset_path, "target": target})

    async def run_optimize(self, model_path: str, dataset_path: str, target: Optional[str]):
        """Optional manual/hyperparameter optimization via OptimizerAgent."""
        if not self.opt_agent:
            logger.info("OptimizerAgent missing — skipping optimize")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.opt_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {"model_path": model_path, "dataset_path": dataset_path, "target": target})

    async def run_model_explain(self, model_path, dataset_path, target):
        if not self.explainer_agent:
            logger.info("ModelExplainerAgent missing — skipping explainability")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.explainer_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {"model_path": model_path, "dataset_path": dataset_path, "target": target})

    async def run_insight_report(self, eda_path, model_summary_path):
        if not self.report_agent:
            logger.info("InsightReporterAgent missing — skipping")
            return None
        # Guard: never pass a .joblib here
        if model_summary_path and model_summary_path.endswith(".joblib"):
            raise ValueError(f"Refusing to parse binary model file as JSON: {model_summary_path}")
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.report_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {"eda_path": eda_path, "model_path": model_summary_path})

    async def run_dashboard(self, eda_path, model_path, insight_path):
        if not self.dashboard_agent:
            logger.info("DashboardAgent missing — skipping")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.dashboard_agent.process, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, {
            "eda_summary_path": eda_path,
            "model_summary_path": model_path,
            "insight_report_path": insight_path,
            "explainability_dir": self.cfg["explain_dir"],
            "visuals_dir": self.cfg["visuals_dir"],
            "output_dashboard_path": "dashboard/app.py"
        })

    async def run_deploy(self, model_meta):
        if not self.deployment_agent:
            logger.info("DeploymentAgent missing — skipping deployment scaffold generation")
            return None
        loop = asyncio.get_event_loop()
        fn = retry_sync(self.deployment_agent.generate_rest_scaffold, attempts=self.cfg["retry_attempts"], base_delay=self.cfg["retry_base_delay"])
        return await run_blocking(loop, fn, model_meta)

    # ---------- Pipeline ----------
    async def run_pipeline(self, dataset_path: str, target: Optional[str] = None, steps: Optional[List[str]] = None):
        steps_to_run = steps or self.cfg["run_steps"]
        logger.info(f"Starting pipeline for dataset={dataset_path} steps={steps_to_run}")

        results: Dict[str, Any] = {}
        eda_path = self.cfg["eda_path"]
        model_pkl_or_joblib_path: Optional[str] = None
        model_summary_path: Optional[str] = None
        insight_report_path: Optional[str] = None

        # 1) EDA
        if "eda" in steps_to_run:
            eda_summary = await self.run_eda(dataset_path)
            results["eda_summary"] = eda_summary
            if eda_summary:
                save_json(eda_summary, eda_path)

        # 2) Feature Engineering
        fe_dataset = dataset_path
        if "feature_engineer" in steps_to_run:
            fe_result = await self.run_feature_engineer(dataset_path)
            results["feature_engineer"] = fe_result
            fe_dataset = _extract_processed_path(fe_result, dataset_path)

        # 3) Model Build
        if "model_build" in steps_to_run:
            model_result = await self.run_model_build(fe_dataset, target)
            results["model_build"] = model_result
            if isinstance(model_result, dict):
                # model may be saved as .joblib or .pkl
                model_pkl_or_joblib_path = model_result.get("model_path") or model_result.get("model_file")
                # find *_summary.json safely
                model_summary_path = _detect_model_summary_path(model_pkl_or_joblib_path, self.cfg["model_dir"])

        # 4) Manual Optimize (optional)
        if "optimize" in steps_to_run and model_pkl_or_joblib_path:
            opt_result = await self.run_optimize(model_pkl_or_joblib_path, fe_dataset, target)
            results["optimize"] = opt_result
            if isinstance(opt_result, dict):
                model_pkl_or_joblib_path = opt_result.get("optimized_model_path", model_pkl_or_joblib_path)
                # try detect summary again if optimizer replaced model
                model_summary_path = _detect_model_summary_path(model_pkl_or_joblib_path, self.cfg["model_dir"]) or model_summary_path

        # 5) Model Explain
        if "model_explain" in steps_to_run and model_pkl_or_joblib_path:
            explain = await self.run_model_explain(model_pkl_or_joblib_path, fe_dataset, target)
            results["model_explain"] = explain

        # 6) Insight Reporter (must pass *_summary.json)
        if "insight_report" in steps_to_run:
            if not model_summary_path:
                # As a last resort, scan model_dir for newest *_summary.json
                model_summary_path = _detect_model_summary_path(None, self.cfg["model_dir"])
            if model_summary_path and Path(model_summary_path).exists():
                report = await self.run_insight_report(eda_path, model_summary_path)
                results["insight_report"] = report
                # Prefer the JSON the reporter writes (if it returns paths)
                if isinstance(report, dict):
                    insight_report_path = report.get("report_json") or report.get("insights_json")
            else:
                logger.info("No model summary JSON available for InsightReporter; skipping insight_report step.")

        # 7) Dashboard
        if "dashboard" in steps_to_run:
            dash = await self.run_dashboard(eda_path, model_summary_path, insight_report_path)
            results["dashboard"] = dash

        # 8) Deploy
        if "deploy" in steps_to_run and model_pkl_or_joblib_path:
            meta = {
                "model_id": Path(model_pkl_or_joblib_path).stem,
                "model_name": Path(model_pkl_or_joblib_path).stem,
                "model_path": model_pkl_or_joblib_path
            }
            deploy = await self.run_deploy(meta)
            results["deploy"] = deploy

        logger.info("✅ Pipeline completed.")
        return results


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AsyncOrchestrator pipeline")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--target",  "-t", default=None)
    parser.add_argument("--steps",   "-s", nargs="*")
    args = parser.parse_args()

    orch = AsyncOrchestrator()

    async def main():
        res = await orch.run_pipeline(args.dataset, args.target, steps=args.steps)
        print(json.dumps(res, indent=2, default=str))

    asyncio.run(main())
