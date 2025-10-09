# backend/hybrid_extractor.py
"""
Hybrid table extractor:
 - Uses pdfplumber for native (text-based) PDFs
 - Falls back to OpenCV + Tesseract OCR for scanned PDFs/images
 - Produces aligned JSON + HTML outputs and a debug overlay (base64 PNG)
 - Returns per-page method and diagnostic logs
"""

from io import BytesIO
import base64
import logging
import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import pdfplumber
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF (fast text detection)

logger = logging.getLogger("hybrid_extractor")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Utilities
# -----------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        pages = [p.convert("RGB") for p in pages]
        return pages
    except Exception as e:
        logger.exception("pdf_to_images conversion failed: %s", e)
        return []


def has_page_text(pdf_bytes: bytes, min_chars_per_page: int = 50) -> List[bool]:
    """
    Check each page of PDF for extractable text using PyMuPDF.
    Returns a list of booleans per page: True if page has vector text above threshold.
    """
    res = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            txt = page.get_text("text") or ""
            res.append(len(txt.strip()) >= min_chars_per_page)
        return res
    except Exception as e:
        logger.exception("has_page_text failed: %s", e)
        return []


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).replace("\x0c", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    # remove obvious OCR junk tokens
    if s.lower() in {"me", "mr", "we", "m r", "lh"} and len(s) <= 3:
        return ""
    return s


# -----------------------
# pdfplumber path (for native PDFs)
# -----------------------
def extract_with_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    results = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_tables = []
                try:
                    tbls = page.extract_tables()
                except Exception as e:
                    logger.info("pdfplumber page %d extract_tables error: %s", i, e)
                    tbls = []
                for t in tbls:
                    # t is list-of-lists
                    df = pd.DataFrame(t)
                    df = df.fillna("").applymap(clean_text)
                    html = df.to_html(index=False, header=False, border=1, justify="left")
                    page_tables.append({"page": i, "method": "pdfplumber", "data": t, "html": html})
                results.append(page_tables)
        # flatten: results is list of lists (one per page)
        return results
    except Exception as e:
        logger.exception("pdfplumber extraction failed: %s", e)
        return []


# -----------------------
# OCR-based (OpenCV + per-cell Tesseract)
# -----------------------
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_base64_png(img_cv) -> str:
    _, buf = cv2.imencode(".png", img_cv)
    return base64.b64encode(buf).decode("ascii")


def preprocess_for_lines(img_cv: np.ndarray, scale_up_small: bool = True):
    """
    Prepare masks to detect horizontal & vertical lines for table detection.
    Returns dict with grayscale, thresh, horiz, vert, scale.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1.0
    if scale_up_small and max(h, w) < 1000:
        scale = 2.0
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # denoise slightly
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # adaptive threshold (invert for morphology)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 9)

    # kernels sized relative to image dims (robust across sizes)
    kernel_len_h = max(10, int(w / 40))
    kernel_len_v = max(10, int(h / 60))

    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v)))

    return {"gray": gray, "thresh": thr, "horiz": horiz, "vert": vert, "scale": scale}


def detect_cell_boxes(horiz_mask, vert_mask, min_area: int = 200):
    """
    Use morphological masks to find contour boxes (candidate cells).
    Returns sorted boxes (x,y,w,h).
    """
    combined = cv2.add(horiz_mask, vert_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        # pad slightly
        pad_x = int(max(2, w * 0.02))
        pad_y = int(max(2, h * 0.02))
        boxes.append((max(0, x - pad_x), max(0, y - pad_y), w + 2 * pad_x, h + 2 * pad_y))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def cluster_column_centers(boxes: List[tuple], x_tol: int = 30) -> List[int]:
    centers = sorted([x + w / 2.0 for x, y, w, h in boxes])
    if not centers:
        return []
    clusters = [[centers[0]]]
    for c in centers[1:]:
        if abs(c - clusters[-1][-1]) <= x_tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    centers_final = [int(sum(g) / len(g)) for g in clusters]
    return centers_final


def ocr_crop_text_and_conf(crop_bgr: np.ndarray) -> (str, int):
    """
    Use pytesseract.image_to_data to get text parts and approximate confidence.
    Returns (text, avg_confidence)
    """
    try:
        data = pytesseract.image_to_data(crop_bgr, output_type=Output.DICT, config="--psm 6")
    except Exception as e:
        logger.exception("pytesseract.image_to_data failed: %s", e)
        return "", -1
    n = len(data.get("text", []))
    texts = []
    confs = []
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        texts.append(t)
        try:
            c = int(data.get("conf", [-1])[i])
        except Exception:
            c = -1
        confs.append(c)
    txt = " ".join(texts).strip()
    avg_conf = int(sum([c for c in confs if c >= 0]) / len([c for c in confs if c >= 0])) if any(c >= 0 for c in confs) else -1
    return clean_text(txt), avg_conf


def ocr_grid_from_boxes(img_cv: np.ndarray, boxes: List[tuple], col_centers: List[int]):
    """
    Given candidate boxes and determined column centers, create a grid where
    each box is assigned to a nearest column. OCR each box and build DataFrame.
    Returns (df, meta_list) where meta_list contains per-cell conf & bbox.
    """
    if not boxes:
        return pd.DataFrame(), []

    # group boxes into rows by y coordinate
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    current = []
    last_y = None
    # y tolerance derived from median box height
    heights = [h for (_, _, _, h) in boxes_sorted] if boxes_sorted else [20]
    median_h = int(np.median(heights)) if heights else 20
    y_tol = max(10, int(median_h * 0.6))

    for (x, y, w, h) in boxes_sorted:
        if last_y is None:
            current = [(x, y, w, h)]
            last_y = y
            continue
        if abs(y - last_y) <= y_tol:
            current.append((x, y, w, h))
            last_y = (last_y + y) // 2
        else:
            rows.append(sorted(current, key=lambda b: b[0]))
            current = [(x, y, w, h)]
            last_y = y
    if current:
        rows.append(sorted(current, key=lambda b: b[0]))

    num_cols = max(1, len(col_centers)) if col_centers else max(1, max(len(r) for r in rows))
    grid = []
    meta = []
    for r in rows:
        row_cells = [""] * num_cols
        row_meta = [None] * num_cols
        for (x, y, w, h) in r:
            cx = x + w / 2.0
            if col_centers:
                col_idx = int(np.argmin([abs(cx - cc) for cc in col_centers]))
            else:
                # fallback: assign by order
                col_idx = min(num_cols - 1, int(round(x / (img_cv.shape[1] / num_cols))))
            # crop, upscale small crops
            ex, ey = max(0, x - 2), max(0, y - 2)
            ew, eh = min(img_cv.shape[1] - ex, w + 4), min(img_cv.shape[0] - ey, h + 4)
            crop = img_cv[ey:ey + eh, ex:ex + ew]
            ch, cw = crop.shape[:2]
            if max(ch, cw) < 60:
                crop = cv2.resize(crop, (int(cw * 2.0), int(ch * 2.0)), interpolation=cv2.INTER_LINEAR)
            text, conf = ocr_crop_text_and_conf(crop)
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + text
            else:
                row_cells[col_idx] = text
            row_meta[col_idx] = {"bbox": (ex, ey, ew, eh), "conf": conf}
        grid.append(row_cells)
        meta.append(row_meta)

    df = pd.DataFrame(grid).fillna("").applymap(lambda s: clean_text(s))
    return df, meta


def make_debug_overlay(img_cv: np.ndarray, boxes: List[tuple], col_centers: List[int]):
    vis = img_cv.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)
    for cc in col_centers:
        cv2.line(vis, (int(cc), 0), (int(cc), vis.shape[0]), (0, 200, 0), 1)
    return cv2_to_b64_png(vis)


def cv2_to_b64_png(img_cv):
    _, buf = cv2.imencode(".png", img_cv)
    return base64.b64encode(buf).decode("ascii")


# -----------------------
# High-level OCR fallback for page image
# -----------------------
def extract_tables_from_image_bytes(image_bytes: bytes, debug: bool = False, dpi_hint: int = 300):
    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_cv = pil_to_cv2(pil)
    prep = preprocess_for_lines(img_cv)
    boxes = detect_cell_boxes(prep["horiz"], prep["vert"], min_area=200)
    if not boxes:
        # fallback: connected components on threshold
        thr = prep["thresh"]
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 300:
                continue
            boxes.append((x, y, w, h))
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    col_centers = cluster_column_centers(boxes, x_tol=max(20, int(pil.size[0] / 60)))
    df, meta = ocr_grid_from_boxes(img_cv, boxes, col_centers)
    # Postprocess header combining
    header_combined = False
    if df.shape[0] >= 2:
        # heuristic: if first two rows both have non-empty entries, combine
        first_nonnull = df.iloc[0].astype(bool).sum()
        second_nonnull = df.iloc[1].astype(bool).sum()
        if first_nonnull and second_nonnull >= first_nonnull:
            hdr = []
            for c in range(df.shape[1]):
                a = df.iat[0, c] if str(df.iat[0, c]).strip() else ""
                b = df.iat[1, c] if str(df.iat[1, c]).strip() else ""
                combined = " / ".join([x for x in (a, b) if x])
                hdr.append(clean_text(combined))
            # drop first two rows and insert combined
            df = df.drop(index=[0, 1]).reset_index(drop=True)
            df.columns = range(df.shape[1])
            df_header = pd.DataFrame([hdr], columns=range(len(hdr)))
            df = pd.concat([df_header, df], ignore_index=True, axis=0)
            header_combined = True

    df = df.replace(r'^\s*$', "", regex=True)
    html = df.to_html(index=False, header=True, border=1, justify="left")
    debug_img_b64 = None
    if debug:
        debug_img_b64 = make_debug_overlay(img_cv, boxes, col_centers)
    return {"method": "ocr", "html": html, "data": df.to_dict(orient="records"), "debug_image": debug_img_b64, "header_combined": header_combined, "boxes": len(boxes), "cols": len(col_centers), "rows": df.shape[0]}


# -----------------------
# Main hybrid entry point
# -----------------------
def extract_tables(pdf_or_image_bytes: bytes, filename: str, debug: bool = False, dpi: int = 300) -> Dict[str, Any]:
    """
    Auto-detects file type and returns a structure:
    {
      pages: [
        { page: int, method: 'pdfplumber'|'ocr', html: str, data: [rows], debug_image: base64 or None, boxes, cols, rows }
      ],
      summary: {...}
    }
    """
    logger.info("extract_tables called for %s (debug=%s)", filename, debug)
    pages_out = []
    summary = {"pages": 0, "methods": {}}

    try:
        if filename.lower().endswith(".pdf"):
            text_flags = has_page_text(pdf_or_image_bytes, min_chars_per_page=60)
            pages = []
            if any(text_flags):
                # try pdfplumber for pages with text
                try:
                    with pdfplumber.open(BytesIO(pdf_or_image_bytes)) as pdf:
                        n_pages = len(pdf.pages)
                        for i, page in enumerate(pdf.pages, start=1):
                            try:
                                tbls = page.extract_tables()
                            except Exception as e:
                                logger.info("pdfplumber page %d extract_tables error: %s", i, e)
                                tbls = []
                            if tbls:
                                # one or more tables on this page
                                for t in tbls:
                                    df = pd.DataFrame(t).fillna("").applymap(clean_text)
                                    pages_out.append({"page": i, "method": "pdfplumber", "html": df.to_html(index=False, header=False, border=1), "data": df.to_dict(orient="records"), "debug_image": None, "boxes": 0, "cols": df.shape[1], "rows": df.shape[0]})
                                summary["methods"].setdefault("pdfplumber", 0)
                                summary["methods"]["pdfplumber"] += 1
                            else:
                                # fallback OCR for this page (convert to image)
                                logger.info("pdfplumber found no table on page %d -> fallback OCR", i)
                                pages.append(i)
                except Exception as e:
                    logger.exception("pdfplumber overall failed: %s", e)
                    pages = list(range(1, len(text_flags) + 1))
            else:
                # no text flags (likely scanned) -> OCR all pages
                pages = "all"

            # process OCR for pages list or all pages
            if pages == "all":
                pil_pages = pdf_to_images(pdf_or_image_bytes, dpi=dpi)
                page_indices = list(range(1, len(pil_pages) + 1))
            elif isinstance(pages, list):
                # need to convert those specific pages to images
                pil_pages = pdf_to_images(pdf_or_image_bytes, dpi=dpi)
                page_indices = pages
            else:
                pil_pages = pdf_to_images(pdf_or_image_bytes, dpi=dpi)
                page_indices = list(range(1, len(pil_pages) + 1))

            # run OCR extraction on page_indices (1-based)
            for pi in page_indices:
                try:
                    pil_img = pil_pages[pi - 1]
                except Exception:
                    continue
                out = extract_tables_from_image_bytes(pil_img.tobytes(), debug=debug, dpi_hint=dpi) if isinstance(pil_img, Image.Image) else extract_tables_from_image_bytes(pil_img.tobytes(), debug=debug, dpi_hint=dpi)
                pages_out.append({"page": pi, **out})
                summary["methods"].setdefault(out.get("method", "ocr"), 0)
                summary["methods"][out.get("method", "ocr")] += 1
        else:
            # single image file
            out = extract_tables_from_image_bytes(pdf_or_image_bytes, debug=debug, dpi_hint=dpi)
            pages_out.append({"page": 1, **out})
            summary["methods"].setdefault(out.get("method", "ocr"), 0)
            summary["methods"][out.get("method", "ocr")] += 1

        summary["pages"] = len(pages_out)
        return {"pages": pages_out, "summary": summary}
    except Exception as e:
        logger.exception("extract_tables failed: %s", e)
        return {"pages": [], "summary": summary, "error": str(e)}




