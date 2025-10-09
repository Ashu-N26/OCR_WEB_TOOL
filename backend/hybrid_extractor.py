"""
backend/hybrid_extractor.py

Hybrid table extractor:
 - Auto-detects section keywords (default: "2.14")
 - For PDFs: uses PyMuPDF to find keyword positions; converts matching page(s) to image
 - For scanned PDFs / images: runs Tesseract (image_to_data) to find keyword positions
 - Crops a region around the found keyword, runs OpenCV table detection in that crop
 - OCRs each detected cell and returns JSON + HTML + debug overlay (base64 PNG)
"""

from io import BytesIO
import base64
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pdfplumber
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import pandas as pd

logger = logging.getLogger("hybrid_extractor")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -------------------------
# Utilities
# -------------------------
def _img_to_b64png(img_cv: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_cv)
    return base64.b64encode(buf).decode("ascii")


def _pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\x0c", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    if len(s) <= 2 and s.lower() in {"me", "mr", "we", "lh"}:
        return ""
    return s


# -------------------------
# Search for keywords in PDF (PyMuPDF)
# -------------------------
def find_keyword_in_pdf_pages(pdf_bytes: bytes, keywords: List[str]) -> List[Tuple[int, fitz.Rect]]:
    """
    Search each page for any of the keywords.
    Returns list of tuples (page_index_1_based, bbox_rect) for each found occurrence (first occurrence per page).
    """
    found = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.exception("Failed to open PDF with PyMuPDF: %s", e)
        return found

    for i, page in enumerate(doc, start=1):
        # try searching for the numeric keyword '2.14' first (common exact match)
        page_found = False
        for kw in keywords:
            try:
                rects = page.search_for(kw, hit_max=16)  # returns list of fitz.Rect
            except Exception:
                rects = []
            if rects:
                # take first match on page
                found.append((i, rects[0]))
                page_found = True
                break
        if page_found:
            continue

        # If none found by search_for (case-sensitive), do a text fallback: lowercased search
        try:
            text = page.get_text("text") or ""
            text_low = text.lower()
            for kw in keywords:
                if kw.lower() in text_low:
                    # We don't have exact bbox; try to search_for again with exact substring or use simple heuristic:
                    try:
                        rects2 = page.search_for(kw)
                        if rects2:
                            found.append((i, rects2[0]))
                        else:
                            # fallback: choose top-of-page bbox (0..page height) near where the text occurs
                            found.append((i, fitz.Rect(0, 0, page.rect.width, page.rect.height)))
                    except Exception:
                        found.append((i, fitz.Rect(0, 0, page.rect.width, page.rect.height)))
                    break
        except Exception:
            continue

    doc.close()
    return found


# -------------------------
# OCR-based keyword detection for images
# -------------------------
def find_keyword_in_image(pil_image: Image.Image, keywords: List[str]) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    Run pytesseract.image_to_data on PIL image and search words for keywords.
    Returns list of tuples: ((x, y, w, h), matched_keyword)
    Coordinates are in pixels (image).
    """
    cv_img = _pil_to_cv(pil_image)
    try:
        data = pytesseract.image_to_data(cv_img, output_type=Output.DICT, config="--psm 6")
    except Exception as e:
        logger.exception("pytesseract.image_to_data error: %s", e)
        return []

    matches = []
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        for kw in keywords:
            # exact substring (case-insensitive)
            if kw.lower() in word.lower():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                matches.append(((x, y, w, h), kw))
    return matches


# -------------------------
# Convert pdf page to PIL image (one page)
# -------------------------
def pdf_page_to_pil(pdf_bytes: bytes, page_number: int, dpi: int = 300) -> Optional[Image.Image]:
    """
    Convert a single PDF page (1-based index) to a PIL image using pdf2image.
    """
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page_number, last_page=page_number)
        if len(imgs) >= 1:
            return imgs[0].convert("RGB")
    except Exception as e:
        logger.exception("pdf2image conversion failed for page %d: %s", page_number, e)
    return None


# -------------------------
# Table detection & OCR from an image region
# -------------------------
def extract_table_from_image_region(pil_img: Image.Image, debug: bool = False) -> Dict[str, Any]:
    """
    Given a PIL image (likely a crop containing table), detect table cells with OpenCV,
    perform per-cell OCR, cluster columns, merge headers, and return JSON + HTML and debug overlay.
    """
    img_cv = _pil_to_cv(pil_img)
    h_img, w_img = img_cv.shape[:2]

    # Preprocess grayscale & threshold
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # adaptive threshold (invert) so lines are white on black
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 9)

    # Morphology to detect lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w_img // 20), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h_img // 40)))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v, iterations=1)

    # Combined mask
    mask = cv2.add(horiz, vert)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 1000:  # skip tiny regions
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        # fallback: try connected components on thr
        contours2, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 1000:
                continue
            boxes.append((x, y, w, h))

    if not boxes:
        # No table-like contours found: do whole-image OCR and return simple one-column result
        try:
            text = pytesseract.image_to_string(img_cv, config="--psm 6")
            rows = [r.strip() for r in text.splitlines() if r.strip()]
            df = pd.DataFrame([[r] for r in rows])
            html = df.to_html(index=False, header=False, border=1)
            return {"method": "ocr_fallback", "data": df.to_dict(orient="records"), "html": html, "debug_image": None}
        except Exception as e:
            logger.exception("OCR fallback failed: %s", e)
            return {"method": "ocr_fallback", "data": [], "html": "", "debug_image": None}

    # Sort boxes top-to-bottom then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Group boxes into rows by y coordinate
    rows_boxes = []
    cur_row = []
    last_y = None
    for (x, y, w, h) in boxes:
        if last_y is None:
            cur_row = [(x, y, w, h)]
            last_y = y
            continue
        if abs(y - last_y) <= max(10, int(h * 0.5)):
            cur_row.append((x, y, w, h))
            last_y = (last_y + y) // 2
        else:
            rows_boxes.append(sorted(cur_row, key=lambda t: t[0]))
            cur_row = [(x, y, w, h)]
            last_y = y
    if cur_row:
        rows_boxes.append(sorted(cur_row, key=lambda t: t[0]))

    # Collect all x-centers for clustering columns
    centers = []
    for row in rows_boxes:
        for (x, y, w, h) in row:
            centers.append(x + w / 2.0)
    centers = sorted(centers)
    # cluster centers with tolerance
    cols = []
    tol = max(20, w_img // 40)
    if centers:
        group = [centers[0]]
        for c in centers[1:]:
            if abs(c - group[-1]) <= tol:
                group.append(c)
            else:
                cols.append(int(sum(group) / len(group)))
                group = [c]
        cols.append(int(sum(group) / len(group)))
    ncols = max(1, len(cols))

    # Build a grid (rows x ncols) and OCR each cell
    grid = []
    meta = []
    for row in rows_boxes:
        row_cells = [""] * ncols
        row_meta = [None] * ncols
        for (x, y, w, h) in row:
            cx = x + w / 2.0
            # find nearest column index
            col_idx = int(np.argmin([abs(cx - c) for c in cols])) if cols else 0
            # crop
            ex, ey = max(0, x - 2), max(0, y - 2)
            ew, eh = min(w + 4, w_img - ex), min(h + 4, h_img - ey)
            crop = img_cv[ey:ey + eh, ex:ex + ew]
            # upscale small crops for better OCR
            ch, cw = crop.shape[:2]
            if max(ch, cw) < 60:
                crop = cv2.resize(crop, (int(cw * 2.0), int(ch * 2.0)), interpolation=cv2.INTER_LINEAR)
            # OCR with confidence
            try:
                data = pytesseract.image_to_data(crop, output_type=Output.DICT, config="--psm 6")
            except Exception:
                # fallback single string
                text = _clean_text(pytesseract.image_to_string(crop, config="--psm 6"))
                row_cells[col_idx] = (row_cells[col_idx] + " " + text).strip() if text else row_cells[col_idx]
                row_meta[col_idx] = {"conf": -1, "bbox": (ex, ey, ew, eh)}
                continue

            parts = []
            confs = []
            for i in range(len(data.get("text", []))):
                t = (data["text"][i] or "").strip()
                if t:
                    parts.append(t)
                    try:
                        c = int(data.get("conf", [-1])[i])
                    except Exception:
                        c = -1
                    confs.append(c)
            text = _clean_text(" ".join(parts))
            avg_conf = int(sum([c for c in confs if c >= 0]) / len([c for c in confs if c >= 0])) if any(c >= 0 for c in confs) else -1
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + text if text else ""
            else:
                row_cells[col_idx] = text
            row_meta[col_idx] = {"conf": avg_conf, "bbox": (ex, ey, ew, eh)}

        grid.append(row_cells)
        meta.append(row_meta)

    # Normalize column counts (pad)
    maxcols = max(len(r) for r in grid) if grid else ncols
    for r in grid:
        while len(r) < maxcols:
            r.append("")

    df = pd.DataFrame(grid).replace(r'^\s*$', "", regex=True)
    # Heuristic: combine first two rows if both appear header-like
    header_combined = False
    if df.shape[0] >= 2:
        first_nonnull = df.iloc[0].astype(bool).sum()
        second_nonnull = df.iloc[1].astype(bool).sum()
        if first_nonnull > 0 and second_nonnull >= first_nonnull:
            hdr = []
            for c in range(df.shape[1]):
                a = df.iat[0, c] if str(df.iat[0, c]).strip() else ""
                b = df.iat[1, c] if str(df.iat[1, c]).strip() else ""
                combined = " / ".join([x for x in (a, b) if x])
                hdr.append(_clean_text(combined))
            df = df.drop(index=[0, 1]).reset_index(drop=True)
            df_header = pd.DataFrame([hdr], columns=range(len(hdr)))
            df = pd.concat([df_header, df], ignore_index=True, axis=0)
            header_combined = True

    html = df.to_html(index=False, header=True, border=1, justify="left")
    debug_img = None
    if debug:
        overlay = img_cv.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 128, 255), 2)
        for cc in cols:
            cv2.line(overlay, (int(cc), 0), (int(cc), overlay.shape[0]), (0, 200, 0), 1)
        debug_img = _img_to_b64png(overlay)

    return {"method": "image_table", "data": df.to_dict(orient="records"), "html": html, "debug_image": debug_img, "rows": df.shape[0], "cols": df.shape[1], "header_combined": header_combined}


# -------------------------
# Top-level extraction routine
# -------------------------
def extract_tables(file_bytes: bytes,
                   filename: str,
                   section_keyword: Optional[str] = None,
                   extra_keywords: Optional[List[str]] = None,
                   debug: bool = False,
                   dpi: int = 300) -> Dict[str, Any]:
    """
    Primary public function.

    - file_bytes: uploaded file bytes
    - filename: original filename (for extension detection)
    - section_keyword: string like "2.14" (defaults to "2.14")
    - extra_keywords: optional list of alternate strings to search (e.g., ["APROXIMAÇÃO","APPROACH"])
    - debug: if True includes debug overlay(s) as base64 PNG
    - dpi: for pdf->image conversion
    """
    if section_keyword is None:
        section_keyword = "2.14"

    keywords = [section_keyword]
    if extra_keywords:
        keywords.extend(extra_keywords)
    # common variants to help detection
    common_variants = [
        "2.14", "2.14 APROXIMAÇÃO", "2.14 APPROACH",
        "APROXIMAÇÃO", "APPROACH", "AD 2.14", "AD 2.14 APROXIMAÇÃO"
    ]
    # add keywords but keep uniqueness and keep user's keyword at front
    for v in common_variants:
        if v not in keywords:
            keywords.append(v)

    filename_lower = filename.lower() if filename else ""
    ext = filename_lower.split(".")[-1] if "." in filename_lower else ""

    results_pages = []
    summary = {"pages": 0, "matches": 0, "method_used": {}}

    # --- PDF path ---
    if ext == "pdf":
        # try PyMuPDF text search first (fast for vector PDFs)
        try:
            found = find_keyword_in_pdf_pages(file_bytes, keywords)
        except Exception as e:
            logger.exception("find_keyword_in_pdf_pages crashed: %s", e)
            found = []

        if found:
            # For each matched page, convert page to image and crop around bbox
            for page_num, rect in found:
                # convert that page to PIL image
                pil_page = pdf_page_to_pil(file_bytes, page_num, dpi=dpi)
                if pil_page is None:
                    continue
                # Map fitz.Rect points to pixels: points * dpi / 72
                # FitZ rect: (x0, y0, x1, y1)
                scale = dpi / 72.0
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                px0 = int(max(0, x0 * scale))
                py0 = int(max(0, y0 * scale))
                px1 = int(min(pil_page.width, x1 * scale))
                py1 = int(min(pil_page.height, y1 * scale))

                # Expand crop to include surrounding table area
                pad_w = int(pil_page.width * 0.05)
                pad_h = int(pil_page.height * 0.18)
                crop_x0 = max(0, px0 - pad_w)
                crop_y0 = max(0, py0 - pad_h)
                crop_x1 = min(pil_page.width, px1 + pad_w)
                crop_y1 = min(pil_page.height, py1 + pad_h)
                crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))

                out = extract_table_from_image_region(crop, debug=debug)
                out.update({"page": page_num, "search_bbox_points": (x0, y0, x1, y1), "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
                results_pages.append(out)
                summary["matches"] += 1
                summary["method_used"].setdefault(out["method"], 0)
                summary["method_used"][out["method"]] += 1

            summary["pages"] = len(results_pages)
            return {"pages": results_pages, "summary": summary}
        else:
            # No exact section found via text layer -> fallback: try OCR on pages and search for keywords
            logger.info("No keyword found in PDF via text layer; falling back to OCR page-by-page.")
            # convert whole PDF to images
            try:
                pil_pages = convert_from_bytes(file_bytes, dpi=dpi)
            except Exception as e:
                logger.exception("pdf2image failed on full conversion: %s", e)
                return {"pages": [], "summary": {"error": str(e)}}

            for page_idx, pil_page in enumerate(pil_pages, start=1):
                matches = find_keyword_in_image(pil_page, keywords)
                if not matches:
                    continue
                # choose first match
                (x, y, w, h), matched_kw = matches[0]
                # expand region
                pad_w = int(pil_page.width * 0.05)
                pad_h = int(pil_page.height * 0.18)
                crop_x0 = max(0, x - pad_w)
                crop_y0 = max(0, y - pad_h)
                crop_x1 = min(pil_page.width, x + w + pad_w)
                crop_y1 = min(pil_page.height, y + h + pad_h)
                crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                out = extract_table_from_image_region(crop, debug=debug)
                out.update({"page": page_idx, "matched_keyword": matched_kw, "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
                results_pages.append(out)
                summary["matches"] += 1
                summary["method_used"].setdefault(out["method"], 0)
                summary["method_used"][out["method"]] += 1

            if results_pages:
                summary["pages"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}

            # last-resort: try pdfplumber's extract_tables on all pages and return tables that contain the keyword text
            logger.info("No OCR keyword hits; last-resort using pdfplumber.extract_tables to search contained text.")
            try:
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        try:
                            tables = page.extract_tables()
                        except Exception:
                            tables = []
                        for t in tables:
                            # check if any cell contains the keyword
                            matched = False
                            for row in t:
                                for cell in row:
                                    if not cell:
                                        continue
                                    for kw in keywords:
                                        if kw.lower() in str(cell).lower():
                                            matched = True
                                            break
                                if matched:
                                    break
                            if matched:
                                df = pd.DataFrame(t).fillna("").applymap(_clean_text)
                                results_pages.append({"page": i, "method": "pdfplumber_table", "data": df.to_dict(orient="records"), "html": df.to_html(index=False, header=False, border=1)})
                                summary["matches"] += 1
                                summary["method_used"].setdefault("pdfplumber_table", 0)
                                summary["method_used"]["pdfplumber_table"] += 1
                summary["pages"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}
            except Exception as e:
                logger.exception("pdfplumber fallback failed: %s", e)
                return {"pages": [], "summary": {"error": str(e)}}

    # --- Image path (or unknown extension) ---
    try:
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        logger.exception("Cannot open uploaded file as image: %s", e)
        return {"pages": [], "summary": {"error": "cannot open file"}}

    matches = find_keyword_in_image(pil_img, keywords)
    if matches:
        # take first match
        (x, y, w, h), matched_kw = matches[0]
        pad_w = int(pil_img.width * 0.05)
        pad_h = int(pil_img.height * 0.18)
        crop_x0 = max(0, x - pad_w)
        crop_y0 = max(0, y - pad_h)
        crop_x1 = min(pil_img.width, x + w + pad_w)
        crop_y1 = min(pil_img.height, y + h + pad_h)
        crop = pil_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        out = extract_table_from_image_region(crop, debug=debug)
        out.update({"page": 1, "matched_keyword": matched_kw, "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
        results_pages.append(out)
        summary["matches"] += 1
        summary["method_used"].setdefault(out["method"], 0)
        summary["method_used"][out["method"]] += 1
        summary["pages"] = len(results_pages)
        return {"pages": results_pages, "summary": summary}

    # If nothing matched:
    logger.info("No section keywords found in image using OCR.")
    # fallback: run full-image table extraction
    out = extract_table_from_image_region(pil_img, debug=debug)
    out.update({"page": 1, "matched_keyword": None})
    results_pages.append(out)
    summary["matches"] += 0
    summary["pages"] = 1
    summary["method_used"].setdefault(out["method"], 0)
    summary["method_used"][out["method"]] += 1
    return {"pages": results_pages, "summary": summary}
