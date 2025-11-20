"""
FinalPDFReportAgent — Fully corrected & hardened single-file implementation
- Robust image sanitization with Pillow (safe defaults, DPI fix, dedupe)
- Explicit width+height provided to ReportLab images (prevents LayoutError)
- Graceful fallback for extremely tall images: place on own page
- Table column widths computed from page size & margins
- Safer dataset preview handling and metrics table sizing
- Detailed logging and clean-up of temp images

Usage: Put this file into your project (next to original), run with Python 3.8+.
Requires: reportlab, pillow, pandas (optional), requests (optional)
"""

import os
import sys
import json
import time
import glob
import shutil
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

# Data libs
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# Imaging
try:
    from PIL import Image as PILImage, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# HTTP for LLM (optional)
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("FinalPDFReportAgent")

# ---------------------------
# Config
# ---------------------------
@dataclass
class FinalReportConfig:
    output_dir: str = "reports"
    llm_provider: str = "openrouter"
    llm_model: str = "deepseek/deepseek-chat"
    max_tokens: int = 1200
    temperature: float = 0.2
    timeout_sec: int = 60
    page_size: Tuple[float, float] = A4
    margin: float = 1.5 * cm
    image_max_width_cm: float = 12.0
    image_max_height_cm: float = 7.0
    image_columns: int = 2
    include_explain_images: bool = True
    include_metrics_table: bool = True
    include_dataset_preview: bool = True
    dataset_preview_rows: int = 8

# ---------------------------
# Helpers
# ---------------------------

def project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def abs_path(rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.abspath(os.path.join(project_root(), rel))

# ---------------------------
# File scanning
# ---------------------------

def scan_for_files(base_dirs: List[str], extensions: Tuple[str, ...]) -> List[str]:
    found = []
    for d in base_dirs:
        d_abs = abs_path(d)
        if not os.path.exists(d_abs):
            continue
        for root, _, files in os.walk(d_abs):
            for f in files:
                if f.lower().endswith(extensions):
                    found.append(os.path.join(root, f))
    found = sorted(set(found), key=os.path.getmtime)
    return found


def find_latest_by_keyword(keywords: List[str], search_dirs: List[str], exts: Tuple[str, ...]) -> Optional[str]:
    files = scan_for_files(search_dirs, exts)
    matches = []
    for f in files:
        name = os.path.basename(f).lower()
        if any(k.lower() in name for k in keywords):
            matches.append(f)
    return matches[-1] if matches else None


def find_summary_json(search_dirs: List[str]) -> Optional[str]:
    files = scan_for_files(search_dirs, (".json",))
    summaries = [f for f in files if "_summary.json" in f.lower() or "summary" in os.path.basename(f).lower()]
    return summaries[-1] if summaries else None


def find_explain_json(search_dirs: List[str]) -> Optional[str]:
    return find_latest_by_keyword(["explain", "explainability"], search_dirs, (".json",))


def find_eda_json(search_dirs: List[str]) -> Optional[str]:
    return find_latest_by_keyword(["eda", "eda_summary"], search_dirs, (".json",))


def find_images(search_dirs: List[str]) -> List[str]:
    imgs = scan_for_files(search_dirs, (".png", ".jpg", ".jpeg"))
    keywords = ["shap", "pdp", "importance", "feature", "explain", "analysis"]
    relevant = [p for p in imgs if any(k in os.path.basename(p).lower() for k in keywords)]
    return relevant if relevant else imgs


def find_models(search_dirs: List[str]) -> List[str]:
    return scan_for_files(search_dirs, (".joblib", ".pkl", ".pt", ".sav", ".model"))


def find_processed_dataset(search_dirs: List[str]) -> Optional[str]:
    files = scan_for_files(search_dirs, (".csv", ".parquet", ".feather"))
    if not files:
        return None
    for name in reversed(files):
        bn = os.path.basename(name).lower()
        if "processed" in bn or "dataset" in bn:
            return name
    return files[-1]

# ---------------------------
# Image sanitizer
# ---------------------------

def sanitize_and_copy_image(src_path: str, tmp_dir: str, max_w_px: int, max_h_px: int) -> Optional[str]:
    try:
        if not os.path.exists(src_path):
            return None
        base = os.path.basename(src_path)
        safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
        dst = os.path.join(tmp_dir, safe_name + ".png")

        if not PIL_AVAILABLE:
            try:
                shutil.copy2(src_path, dst)
                return dst
            except Exception:
                return None

        with PILImage.open(src_path) as im:
            # convert to RGB to avoid palette/alpha surprises
            im = im.convert("RGB")

            # sanitize DPI
            dpi = im.info.get("dpi", (72, 72))
            try:
                if isinstance(dpi, tuple):
                    if dpi[0] <= 0 or dpi[1] <= 0 or dpi[0] > 600 or dpi[1] > 600:
                        dpi = (72, 72)
                else:
                    dpi = (72, 72)
            except Exception:
                dpi = (72, 72)

            # enforce hard pixel limits (to avoid extremely large images)
            w, h = im.size
            if w <= 0 or h <= 0:
                return None

            # if image exceeds limits, downscale preserving aspect ratio
            if w > max_w_px or h > max_h_px:
                im.thumbnail((max_w_px, max_h_px), PILImage.LANCZOS)

            # final containment to be sure
            w2, h2 = im.size
            if w2 > max_w_px or h2 > max_h_px:
                im = ImageOps.contain(im, (max_w_px, max_h_px))

            # save normalized PNG
            im.save(dst, format="PNG", dpi=(72, 72))
            return dst
    except Exception as e:
        log.warning(f"Image sanitizer failed for {src_path}: {e}")
        return None

# ---------------------------
# LLM client (optional)
# ---------------------------
class LLMClient:
    def __init__(self, cfg: FinalReportConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        if self.cfg.llm_provider == "none":
            return False
        if self.cfg.llm_provider == "openrouter":
            return bool(os.getenv("OPENROUTER_API_KEY"))
        if self.cfg.llm_provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        return False

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests not installed")
        if not self.enabled():
            raise RuntimeError("LLM not configured")
        # keep minimal; caller handles exceptions
        if self.cfg.llm_provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            key = os.getenv("OPENROUTER_API_KEY")
            payload = {
                "model": self.cfg.llm_model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature
            }
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_sec)
            r.raise_for_status()
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return str(data)
        elif self.cfg.llm_provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            key = os.getenv("OPENAI_API_KEY")
            payload = {
                "model": self.cfg.llm_model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature
            }
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_sec)
            r.raise_for_status()
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return str(data)
        else:
            raise RuntimeError("Unsupported LLM provider")

# ---------------------------
# PDF helpers
# ---------------------------

def _header_footer(canvas_obj: canvas.Canvas, doc):
    canvas_obj.saveState()
    width, height = doc.pagesize
    canvas_obj.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(doc.leftMargin, height - doc.topMargin + 6, width - doc.rightMargin, height - doc.topMargin + 6)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(width - doc.rightMargin, doc.bottomMargin - 10, f"Page {doc.page}")
    canvas_obj.restoreState()


def bullet_paragraphs(items: List[str], styles):
    return [Paragraph(f"• {it}", styles["BodyText"]) for it in items]

# compute available frame height for images (approx)
def _available_frame_height(cfg: FinalReportConfig):
    page_w, page_h = cfg.page_size
    usable_h = page_h - (cfg.margin * 2)
    # keep some room for header/footer
    usable_h = max(usable_h - (1.0 * cm), 100)
    return usable_h

# ---------------------------
# Main Agent
# ---------------------------
class FinalPDFReportAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = FinalReportConfig()
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)

        self.output_dir = abs_path(self.cfg.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tmp_dir = os.path.join(self.output_dir, "tmp_images")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.llm = LLMClient(self.cfg)
        log.info(f"Initialized FinalPDFReportAgent. Output dir: {self.output_dir}")

    def _load_json(self, path: Optional[str]) -> Dict[str, Any]:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to parse JSON {path}: {e}")
        return {}

    def _load_dataset_preview(self, csv_path: str, nrows: int = 8):
        if not csv_path or not os.path.exists(csv_path) or not PANDAS_AVAILABLE:
            return None
        try:
            df = pd.read_csv(csv_path, nrows=nrows)
            return df
        except Exception as e:
            log.warning(f"Failed to load dataset preview {csv_path}: {e}")
            return None

    def _safe_image_list(self, raw_images: List[str]) -> List[str]:
        seen = set()
        unique = []
        for p in raw_images:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        px_per_cm = 72.0 / 2.54
        max_w_px = int(self.cfg.image_max_width_cm * px_per_cm)
        max_h_px = int(self.cfg.image_max_height_cm * px_per_cm)

        out = []
        for p in unique:
            try:
                sanitized = sanitize_and_copy_image(p, self.tmp_dir, max_w_px, max_h_px)
                if sanitized and os.path.exists(sanitized):
                    out.append(sanitized)
                else:
                    log.warning(f"Skipping invalid image: {p}")
            except Exception as e:
                log.warning(f"Error sanitizing {p}: {e}")
        return out

    def _generate_sections(self, eda: Dict[str, Any], model: Dict[str, Any], explain: Dict[str, Any], insights_text: str) -> Dict[str, Any]:
        if self.llm.enabled():
            try:
                system = "You are a senior data scientist. Produce JSON with: executive_summary, key_findings (list), recommendations (list), limitations (list), next_steps (list)."
                payload = {"eda": eda, "model": model, "explain": explain, "insights": insights_text[:12000]}
                raw = self.llm.complete(system, json.dumps(payload, default=str))
                if isinstance(raw, str) and raw.strip().startswith("{"):
                    try:
                        return json.loads(raw)
                    except Exception:
                        first = raw.find("{"); last = raw.rfind("}")
                        if first != -1 and last != -1:
                            try:
                                return json.loads(raw[first:last+1])
                            except Exception:
                                log.warning("LLM returned non-JSON; falling back.")
            except Exception as e:
                log.warning(f"LLM failed: {e}")

        exec_summary = f"Report generated. Model: {model.get('best_model') or model.get('model_name') or model.get('model','N/A')}. Task: {model.get('task_type', 'Unknown')}."
        findings = []
        if isinstance(eda, dict):
            top_corr = eda.get("top_correlations") or eda.get("correlations") or []
            if top_corr:
                findings.append(f"Top correlations: {top_corr[:5]}")
            mv = eda.get("missing_values") or eda.get("missing_percent")
            if mv:
                findings.append("Missing values summary available.")
        metrics = model.get("metrics") if isinstance(model, dict) else {}
        if metrics:
            findings.append(f"Model metrics: {metrics}")
        if isinstance(explain, dict) and explain.get("model_based_importance"):
            findings.append(f"Top features: {list(explain['model_based_importance'].keys())[:8]}")
        return {
            "executive_summary": exec_summary,
            "key_findings": findings or ["No explicit findings extracted."],
            "recommendations": ["Validate data quality.", "Add hyperparameter optimization and CV.", "Monitor model drift."],
            "limitations": ["Analysis based on dataset snapshot."],
            "next_steps": ["Run optimizer agent; deploy with monitoring."]
        }

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        inp = input_data or {}

        search_dirs = ["data", "data/reports", "data/models", "data/plots", "reports", "plots", "visuals", "."]

        eda_path = find_eda_json(search_dirs)
        model_json_path = find_summary_json(search_dirs)
        explain_json_path = find_explain_json(search_dirs)
        raw_images = []
        model_files = find_models(search_dirs)
        processed_csv = find_processed_dataset(search_dirs)

        log.info(f"Files discovered: eda={eda_path}, model_json={model_json_path}, explain_json={explain_json_path}, processed_csv={processed_csv}")
        log.info(f"Image loading disabled; skipping all images.")

        eda = self._load_json(eda_path)
        model = self._load_json(model_json_path)
        explain = self._load_json(explain_json_path)

        insights_text = ""
        for candidate in ["data/reports/insights_report.txt", "reports/insights_report.txt", "data/reports/insights_report.json", "reports/insights_report.json"]:
            p = abs_path(candidate)
            if os.path.exists(p):
                try:
                    if p.lower().endswith(".json"):
                        j = self._load_json(p)
                        insights_text = j.get("insights_text", "") if isinstance(j, dict) else ""
                    else:
                        with open(p, "r", encoding="utf-8") as f:
                            insights_text = f.read()
                    log.info(f"Loaded insights from {p}")
                    break
                except Exception:
                    continue

        dataset_preview = self._load_dataset_preview(processed_csv, self.cfg.dataset_preview_rows) if processed_csv and PANDAS_AVAILABLE else None

        images_safe = [](raw_images)
        if not images_safe:
            log.info("No valid explainability images after validation.")

        sections = self._generate_sections(eda, model, explain, insights_text)

        sections_path = os.path.join(self.output_dir, "final_report_sections.json")
        try:
            with open(sections_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, indent=2)
            log.info(f"Wrote sections JSON: {sections_path}")
        except Exception as e:
            log.error(f"Failed to write sections JSON: {e}")

        pdf_path = os.path.join(self.output_dir, "final_report.pdf")
        try:
            styles = getSampleStyleSheet()
            if "CustomItalic" not in styles:
                styles.add(ParagraphStyle("CustomItalic", parent=styles["Normal"], fontSize=9, textColor=colors.grey))

            doc = SimpleDocTemplate(pdf_path, pagesize=self.cfg.page_size,
                                    leftMargin=self.cfg.margin, rightMargin=self.cfg.margin,
                                    topMargin=self.cfg.margin, bottomMargin=self.cfg.margin,
                                    title="AI Data Science Report", author="AI Pipeline")

            story: List[Any] = []

            # Title
            story.append(Paragraph("AI Data Science Report", styles["Title"]))
            story.append(Paragraph("Automated Insights, Models, and Explainability", styles["Heading2"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            story.append(PageBreak())

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles["Heading2"]))
            story.append(Paragraph(sections.get("executive_summary", ""), styles["BodyText"]))
            story.append(PageBreak())

            # Key Findings
            story.append(Paragraph("Key Findings", styles["Heading2"]))
            for p in bullet_paragraphs(sections.get("key_findings", []), styles):
                story.append(p)
            story.append(PageBreak())

            # Metrics
            if self.cfg.include_metrics_table and isinstance(model, dict):
                metrics = model.get("metrics") or {}
                if metrics:
                    story.append(Paragraph("Model Metrics", styles["Heading2"]))
                    data = [["Metric", "Value"]]
                    for k, v in metrics.items():
                        data.append([str(k), str(v)])
                    # compute column widths relative to page width
                    page_w, _ = self.cfg.page_size
                    col_w = (page_w - 2 * self.cfg.margin) / 2
                    tbl = Table(data, hAlign="LEFT", colWidths=[col_w, col_w])
                    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                                             ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                             ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
                    story.append(tbl)
                    story.append(PageBreak())
                else:
                    log.info("No metrics present to show in metrics table.")

            # Dataset preview
            if self.cfg.include_dataset_preview and dataset_preview is not None:
                story.append(Paragraph("Dataset Preview (first rows)", styles["Heading2"]))
                df_preview = dataset_preview
                cols = list(df_preview.columns)
                table_data = [cols] + df_preview.head(self.cfg.dataset_preview_rows).astype(str).values.tolist()
                # auto column widths (naive): split equally
                page_w, _ = self.cfg.page_size
                total_width = page_w - 2 * self.cfg.margin
                col_count = max(1, len(cols))
                col_w = total_width / col_count
                tbl = Table(table_data, hAlign="LEFT", colWidths=[col_w] * col_count)
                tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                         ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                                         ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
                story.append(tbl)
                story.append(PageBreak())

            # EDA highlights
            if isinstance(eda, dict) and eda:
                story.append(Paragraph("EDA Highlights", styles["Heading2"]))
                mv = eda.get("missing_values") or eda.get("missing_percent")
                if mv:
                    story.append(Paragraph("Missing value summary:", styles["BodyText"]))
                    story.append(Paragraph(str(mv)[:2000], styles["BodyText"]))
                corr = eda.get("top_correlations") or eda.get("correlations")
                if corr:
                    story.append(Paragraph("Top correlations:", styles["BodyText"]))
                    if isinstance(corr, list):
                        for item in corr[:10]:
                            story.append(Paragraph(str(item), styles["BodyText"]))
                    else:
                        story.append(Paragraph(str(corr), styles["BodyText"]))
                story.append(PageBreak())

            # Explainability visuals disabled
            log.info("Explainability visuals skipped (images disabled).")
            else:
                log.info("No explainability images included or none valid.") (grid) - hardened
            if self.cfg.include_explain_images and images_safe:
                story.append(Paragraph("Explainability Visuals", styles["Heading2"]))

                # target sizes in points
                w_pt = self.cfg.image_max_width_cm * cm
                h_pt = self.cfg.image_max_height_cm * cm

                cols = max(1, int(self.cfg.image_columns))
                page_w, page_h = self.cfg.page_size
                available_h = _available_frame_height(self.cfg)

                # compute table column widths so images fit the page area
                total_width = page_w - 2 * self.cfg.margin
                col_w_pt = total_width / cols

                rows = []
                row = []

                def flush_row_to_rows():
                    nonlocal row
                    if row:
                        while len(row) < cols:
                            row.append(Paragraph("", styles["Normal"]))
                        rows.append(row)
                        row = []

                for idx, img in enumerate(images_safe):
                    try:
                        # read original size for aspect
                        if PIL_AVAILABLE:
                            with PILImage.open(img) as im:
                                orig_w, orig_h = im.size
                                if orig_w <= 0:
                                    raise ValueError("invalid image dimensions")
                                aspect = orig_h / float(orig_w)
                        else:
                            # fallback: assume square
                            orig_w, orig_h = (int(w_pt), int(h_pt))
                            aspect = orig_h / float(orig_w)

                        # compute target size within col width and configured max height
                        target_w = min(col_w_pt - (6 + 6), w_pt)  # subtract paddings
                        target_h = target_w * aspect

                        # cap to configured max height
                        if target_h > h_pt:
                            target_h = h_pt
                            target_w = target_h / aspect

                        # ensure it won't exceed available frame height for a table cell
                        if target_h > available_h:
                            # image is too tall for a table row — place it on its own page instead
                            log.info(f"Image {img} too tall for grid; placing on its own page")
                            # flush any current row first
                            flush_row_to_rows()
                            # add the large image as a standalone block
                            rlimg_large = RLImage(img, width=min(w_pt, total_width), height=None)
                            # ensure explicit dimensions using PIL if available
                            if PIL_AVAILABLE:
                                with PILImage.open(img) as _im:
                                    ow, oh = _im.size
                                    asp = oh / float(ow)
                                    tw = min(w_pt, total_width)
                                    th = tw * asp
                                    if th > available_h:
                                        th = available_h
                                        tw = th / asp
                                    rlimg_large = RLImage(img, width=tw, height=th)
                            story.append(rlimg_large)
                            story.append(Spacer(1, 6))
                            story.append(Paragraph(os.path.basename(img), styles["CustomItalic"]))
                            story.append(PageBreak())
                            continue

                        # normal grid image
                        rlimg = RLImage(img, width=target_w, height=target_h)
                        caption = Paragraph(os.path.basename(img), styles["CustomItalic"]) if os.path.basename(img) else Spacer(1, 1)
                        cell = KeepTogether([rlimg, Spacer(1, 4), caption])
                        row.append(cell)

                    except Exception as e:
                        log.warning(f"Failed to add image {img}: {e}")
                        continue

                    if (idx + 1) % cols == 0:
                        rows.append(row)
                        row = []

                # pad last row
                if row:
                    while len(row) < cols:
                        row.append(Paragraph("", styles["Normal"]))
                    rows.append(row)

                if rows:
                    # set column widths so table cells match allowed column width
                    col_widths = [col_w_pt] * cols
                    table = Table(rows, hAlign="LEFT", colWidths=col_widths)
                    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                               ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                               ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
                    story.append(table)
                    story.append(PageBreak())
            else:
                log.info("No explainability images included or none valid.")

            # Recommendations, Limitations, Next steps
            story.append(Paragraph("Recommendations", styles["Heading2"]))
            for p in bullet_paragraphs(sections.get("recommendations", []), styles):
                story.append(p)
            story.append(Spacer(1, 12))

            story.append(Paragraph("Limitations", styles["Heading2"]))
            for p in bullet_paragraphs(sections.get("limitations", []), styles):
                story.append(p)
            story.append(Spacer(1, 12))

            story.append(Paragraph("Next Steps", styles["Heading2"]))
            for p in bullet_paragraphs(sections.get("next_steps", []), styles):
                story.append(p)
            story.append(PageBreak())

            # Appendix: list of files used
            story.append(Paragraph("Appendix: Files used", styles["Heading2"]))
            files_used = []
            if eda_path: files_used.append(eda_path)
            if model_json_path: files_used.append(model_json_path)
            if explain_json_path: files_used.append(explain_json_path)
            if processed_csv: files_used.append(processed_csv)
            files_used += model_files
            files_used += raw_images
            for f in files_used:
                story.append(Paragraph(f, styles["BodyText"]))

            doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
            elapsed = round(time.time() - start, 2)
            log.info(f"Final PDF written: {os.path.abspath(pdf_path)} (took {elapsed}s)")
        except Exception as e:
            log.error("PDF build failed: " + str(e))
            log.error(traceback.format_exc())
            raise
        finally:
            try:
                if os.path.exists(self.tmp_dir):
                    shutil.rmtree(self.tmp_dir)
                    log.info("Temporary images cleaned up.")
            except Exception as e:
                log.warning(f"Failed to remove temp images: {e}")

        return {
            "pdf_path": os.path.abspath(pdf_path),
            "sections_json": os.path.abspath(sections_path),
            "duration_sec": round(time.time() - start, 2)
        }

# Run standalone
if __name__ == "__main__":
    agent = FinalPDFReportAgent()
    try:
        out = agent.process()
        print(json.dumps(out, indent=2))
    except Exception:
        log.exception("Agent failed:")
        sys.exit(1)
