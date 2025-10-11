# backend/hybrid_extractor.py
"""
Hybrid table extractor (updated, robust).

Features / fixes:
- Accurate page-type detection (text vs scanned image vs mixed)
- Safe section-keyword regex builder (no re.sub replacement escapes)
- Pdf text-layer table reconstruction (PyMuPDF / pdfplumber)
- Image rasterization fallback (PyMuPDF or pdf2image)
- Robust OpenCV grid detection (horizontal + vertical line detection)
- OCR-based cell extraction using pytesseract.image_to_data
- OCR clustering fallback (if no grid lines present)
- Detailed debug info and base64 debug image overlays
- Defensive: will return clear error info if optional packages are missing
"""

from io import BytesIO
import base64
import logging
import re
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher

# Optional imports (defensive)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None
    Output = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger("hybrid_extractor")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Helpers
# -----------------------
def _clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x0c", " ").replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s


def _img_to_b64png(img_cv: "np.ndarray") -> str:
    if cv2 is None:
        return ""
    try:
        _, buf = cv2.imencode(".png", img_cv)
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return ""


def _pil_to_cv(pil_img: "Image.Image"):
    if Image is None or np is None or cv2 is None:
        raise RuntimeError("Pillow/numpy/opencv required for image conversions")
    arr = np.array(pil_img)
    # if grayscale
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.65) -> bool:
    if not haystack or not needle:
        return False
    hay = haystack.lower()
    needle = needle.lower()
    if needle in hay:
        return True
    # token window fuzzy match
    tokens = hay.split()
    n = len(tokens)
    m = len(needle.split())
    if m == 0:
        return False
    for window in range(m, min(n, m + 5) + 1):
        for i in range(0, n - window + 1):
            sub = " ".join(tokens[i:i + window])
            ratio = SequenceMatcher(None, sub, needle).ratio()
            if ratio >= threshold:
                return True
    if SequenceMatcher(None, hay, needle).ratio() >= threshold:
        return True
    return False


# -----------------------
# Build safe section patterns (no bad escape)
# -----------------------
def _make_section_patterns(section_keyword: str) -> List[re.Pattern]:
    base = (section_keyword or "").strip()
    if not base:
        return []
    # Escape characters but replace literal '.' with flexible group (spaces, dots, commas)
    escaped_chars = []
    for ch in base:
        if ch == ".":
            escaped_chars.append(r'[\s\.,]*')
        else:
            escaped_chars.append(re.escape(ch))
    base_fuzzy = "".join(escaped_chars)
    variants = [
        rf"{base_fuzzy}",
        rf"{base_fuzzy}.*approach",
        rf"approach.*{base_fuzzy}",
        rf"{base_fuzzy}.*runway",
        rf"aproxima[cç][aã]o.*{base_fuzzy}",
        rf"{base_fuzzy}.*ilumin",
    ]
    patterns = []
    for v in variants:
        try:
            patterns.append(re.compile(v, re.IGNORECASE))
        except Exception:
            continue
    return patterns


# -----------------------
# Page type detection
# -----------------------
def _detect_page_mode_with_fitz(doc, page_index: int) -> Dict[str, Any]:
    """
    Inspect a page using PyMuPDF (fitz) heuristics and decide:
      - 'vector' (text-dominant)
      - 'image' (image-dominant / scanned)
      - 'mixed'
    Returns dict with mode and some stats.
    """
    res = {"mode": "unknown", "text_blocks": 0, "text_chars": 0, "image_count": 0}
    try:
        page = doc[page_index]
        # words/blocks
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = []
        text_chars = 0
        text_blocks = 0
        for b in blocks:
            # block item is tuple where text is at index 4 for "blocks" or sometimes 4/5; protectively handle
            text = ""
            if isinstance(b, (list, tuple)) and len(b) >= 5:
                text = str(b[4] or "")
            else:
                try:
                    text = page.get_text("text") or ""
                except Exception:
                    text = ""
            if text and text.strip():
                text_blocks += 1
                text_chars += len(text.strip())
        # images
        try:
            images = page.get_images(full=True) or []
        except Exception:
            images = []
        res["text_blocks"] = text_blocks
        res["text_chars"] = text_chars
        res["image_count"] = len(images)
        # heuristics
        if len(images) > 0 and text_chars < 80:
            res["mode"] = "image"
        elif text_chars < 80 and len(images) == 0:
            # very little text and no images - treat as vector text but low density
            res["mode"] = "low_text"
        else:
            # enough text
            res["mode"] = "vector"
    except Exception as e:
        logger.exception("detect_page_mode_with_fitz failed: %s", e)
        res["mode"] = "unknown"
    return res


# -----------------------
# PDF text-layer table reconstruction (PyMuPDF words -> columns)
# -----------------------
def _table_from_words(words: List[Tuple[float, float, float, float, str]]) -> List[List[str]]:
    if not words or np is None:
        return []
    # Sort by y (top) then x (left)
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    # Group into lines by y tolerance
    tol_y = 6.0
    lines = []
    cur = [words_sorted[0]]
    last_y = words_sorted[0][1]
    for w in words_sorted[1:]:
        y = w[1]
        if abs(y - last_y) <= tol_y:
            cur.append(w)
            last_y = (last_y + y) / 2.0
        else:
            lines.append(sorted(cur, key=lambda x: x[0]))
            cur = [w]
            last_y = y
    if cur:
        lines.append(sorted(cur, key=lambda x: x[0]))
    # Build candidate column centers
    centers = []
    for ln in lines:
        for (x0, y0, x1, y1, txt) in ln:
            centers.append((x0 + x1) / 2.0)
    if not centers:
        return []
    centers_sorted = sorted(centers)
    gap_thresh = max(20.0, (max(centers_sorted) - min(centers_sorted)) / 30.0)
    clusters = []
    cluster = [centers_sorted[0]]
    for c in centers_sorted[1:]:
        if abs(c - cluster[-1]) <= gap_thresh:
            cluster.append(c)
        else:
            clusters.append(sum(cluster) / len(cluster))
            cluster = [c]
    clusters.append(sum(cluster) / len(cluster))
    col_centers = clusters
    # Compose rows
    table = []
    for ln in lines:
        row = [""] * len(col_centers)
        for (x0, y0, x1, y1, txt) in ln:
            cx = (x0 + x1) / 2.0
            dists = [abs(cx - cc) for cc in col_centers]
            col_idx = int(min(range(len(dists)), key=lambda i: dists[i])) if dists else 0
            if row[col_idx]:
                row[col_idx] += " " + txt
            else:
                row[col_idx] = txt
        if any(c.strip() for c in row):
            table.append([_clean_text(c) for c in row])
    return table


def _extract_tables_from_pdf_text_layer(pdf_bytes: bytes, section_keyword: str) -> List[Dict[str, Any]]:
    results = []
    if fitz is None:
        return results
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return results
    patterns = _make_section_patterns(section_keyword)
    try:
        for i in range(len(doc)):
            try:
                page = doc[i]
                words = page.get_text("words") or []
                if not words:
                    continue
                page_text = " ".join([w[4] for w in words if len(w) >= 5])
                found_header = any(p.search(page_text) for p in patterns) if patterns else False
                words_tuples = []
                for w in words:
                    # w: (x0, y0, x1, y1, "word", ...)
                    try:
                        if len(w) >= 5:
                            x0, y0, x1, y1, txt = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
                            words_tuples.append((x0, y0, x1, y1, txt))
                    except Exception:
                        continue
                table = _table_from_words(words_tuples)
                if table and len(table) >= 2 and any(len(r) > 1 for r in table):
                    df_html = pd.DataFrame(table).to_html(index=False, header=False, border=1) if pd is not None else None
                    results.append({
                        "page": i + 1,
                        "method": "pdf_text_reconstruct",
                        "data": table,
                        "html": df_html,
                        "found_header": found_header
                    })
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return results


# -----------------------
# Image table detection (OpenCV)
# -----------------------
def _preprocess_for_table(cv_img: "np.ndarray", max_dim: int = 2000) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy required")
    h, w = cv_img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # contrast enhancement + denoise
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold (inverse b/w)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 9)
    return bin_img, gray, cv_img


def _find_table_grid(cv_img: "np.ndarray", debug: bool = False) -> Dict[str, Any]:
    """
    Detect table grid lines and return bounding boxes for cells.
    Returns dict:
       { 'cells': [[(x,y,w,h), ...], ...], 'overlay': overlay_image or None }
    """
    result = {"cells": [], "overlay": None}
    if cv2 is None or np is None:
        return result
    try:
        bin_img, gray, orig = _preprocess_for_table(cv_img)
        h, w = bin_img.shape
        # horizontal and vertical kernels
        horiz_size = max(10, w // 30)
        vert_size = max(10, h // 40)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        # detect lines
        horiz = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        mask = cv2.add(horiz, vert)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        # intersections may indicate cell corners
        joints = cv2.bitwise_and(horiz, vert)
        # find contours on mask (cells clusters)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if ww * hh < 800:  # small noise threshold
                continue
            boxes.append((x, y, ww, hh))
        if not boxes:
            # maybe the lines are broken; try HoughLinesP to detect long lines
            lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=150, minLineLength=max(w, h) // 10, maxLineGap=20)
            # Hough fallback doesn't give boxes directly -> skip advanced fallback here
        else:
            # group boxes into rows by y
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
            rows = []
            current_row = [boxes[0]]
            last_y = boxes[0][1]
            for b in boxes[1:]:
                if abs(b[1] - last_y) <= max(10, int(b[3] * 0.6)):
                    current_row.append(b)
                    last_y = int((last_y + b[1]) / 2)
                else:
                    rows.append(sorted(current_row, key=lambda r: r[0]))
                    current_row = [b]
                    last_y = b[1]
            if current_row:
                rows.append(sorted(current_row, key=lambda r: r[0]))
            # now determine column centers from the largest row
            col_centers = []
            if rows:
                largest_row = max(rows, key=lambda r: len(r))
                centers = [int(x + wbox / 2) for (x, y, wbox, hbox) in largest_row]
                col_centers = sorted(centers)
            # build grid of cell boxes aligned to column centers
            grid = []
            for r in rows:
                row_cells = [""] * max(1, len(col_centers))
                cell_boxes = [None] * max(1, len(col_centers))
                for (x, y, ww, hh) in r:
                    cx = int(x + ww / 2)
                    if not col_centers:
                        col_idx = 0
                    else:
                        col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
                    cell_boxes[col_idx] = (x, y, ww, hh)
                grid.append(cell_boxes)
            result["cells"] = grid
            # debug overlay
            if debug:
                overlay = orig.copy()
                for row in result["cells"]:
                    for box in row:
                        if box is None:
                            continue
                        x, y, ww, hh = box
                        cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 180, 255), 2)
                result["overlay"] = overlay
    except Exception as e:
        logger.exception("find_table_grid failed: %s", e)
    return result


def _ocr_cells_from_boxes(orig_img: "np.ndarray", cell_boxes: List[List[Optional[Tuple[int, int, int, int]]]]) -> List[List[str]]:
    """
    Crop each box and OCR it. Returns list-of-rows with text strings.
    """
    if pytesseract is None or cv2 is None:
        raise RuntimeError("pytesseract/opencv required")
    rows_text = []
    for row in cell_boxes:
        row_texts = []
        for box in row:
            if not box:
                row_texts.append("")
                continue
            x, y, w, h = box
            # small padding
            px = max(0, x - 2)
            py = max(0, y - 2)
            pw = min(w + 4, orig_img.shape[1] - px)
            ph = min(h + 4, orig_img.shape[0] - py)
            crop = orig_img[py:py + ph, px:px + pw]
            try:
                txt = pytesseract.image_to_string(crop, config="--psm 6")
                row_texts.append(_clean_text(txt))
            except Exception:
                row_texts.append("")
        rows_text.append(row_texts)
    return rows_text


# -----------------------
# OCR clustering fallback (no grid lines)
# -----------------------
def _ocr_cluster_table_from_image(orig_img: "np.ndarray", debug: bool = False) -> Dict[str, Any]:
    """
    Use pytesseract.image_to_data to get word boxes and cluster them into rows and columns.
    Returns dict with data list-of-rows and optional overlay.
    """
    res = {"method": "ocr_cluster", "data": [], "overlay": None}
    if pytesseract is None or cv2 is None or np is None:
        res["error"] = "pytesseract/opencv/numpy missing"
        return res
    try:
        odata = pytesseract.image_to_data(orig_img, output_type=Output.DICT, config="--psm 6")
    except Exception as e:
        res["error"] = f"tesseract failed: {e}"
        return res
    words = []
    n = len(odata.get("text", []))
    for i in range(n):
        txt = (odata["text"][i] or "").strip()
        if not txt:
            continue
        try:
            left = int(odata.get("left", [0])[i])
            top = int(odata.get("top", [0])[i])
            width = int(odata.get("width", [0])[i])
            height = int(odata.get("height", [0])[i])
        except Exception:
            continue
        conf = int(odata.get("conf", [-1])[i]) if odata.get("conf") else -1
        words.append({"text": txt, "left": left, "top": top, "width": width, "height": height, "conf": conf})
    if not words:
        res["error"] = "no OCR words"
        return res
    # cluster by y -> rows
    words_sorted = sorted(words, key=lambda w: (w["top"], w["left"]))
    rows = []
    cur = [words_sorted[0]]
    last_y = words_sorted[0]["top"]
    for w in words_sorted[1:]:
        if abs(w["top"] - last_y) <= max(10, int(w["height"] * 0.6)):
            cur.append(w)
            last_y = int(sum([x["top"] for x in cur]) / len(cur))
        else:
            rows.append(sorted(cur, key=lambda x: x["left"]))
            cur = [w]
            last_y = w["top"]
    if cur:
        rows.append(sorted(cur, key=lambda x: x["left"]))
    # compute column centers from all centers
    centers = []
    for r in rows:
        for w in r:
            centers.append(w["left"] + w["width"] / 2.0)
    centers = sorted(centers)
    col_centers = []
    if centers:
        tol = max(20, int(orig_img.shape[1] / 40))
        group = [centers[0]]
        for c in centers[1:]:
            if abs(c - group[-1]) <= tol:
                group.append(c)
            else:
                col_centers.append(int(sum(group) / len(group)))
                group = [c]
        col_centers.append(int(sum(group) / len(group)))
    if not col_centers:
        col_centers = [int(orig_img.shape[1] / 2)]
    # fill grid
    grid = []
    for r in rows:
        row_cells = ["" for _ in range(len(col_centers))]
        for w in r:
            cx = int(w["left"] + w["width"] / 2.0)
            col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + w["text"]
            else:
                row_cells[col_idx] = w["text"]
        grid.append([_clean_text(c) for c in row_cells])
    res["data"] = grid
    if debug:
        overlay = orig_img.copy()
        for w in words:
            cv2.rectangle(overlay, (w["left"], w["top"]), (w["left"] + w["width"], w["top"] + w["height"]), (255, 0, 0), 1)
        for c in col_centers:
            cv2.line(overlay, (int(c), 0), (int(c), overlay.shape[0]), (0, 255, 0), 1)
        res["overlay"] = overlay
    return res


# -----------------------
# Public extract_tables entrypoint (main)
# -----------------------
def extract_tables(file_bytes: bytes,
                   filename: str,
                   section_keyword: str = "2.14",
                   extra_keywords: Optional[List[str]] = None,
                   debug: bool = False,
                   dpi: int = 300) -> Dict[str, Any]:
    """
    Main hybrid extractor - returns {"pages": [...], "summary": {...}}
    Each page entry contains:
      - page (1-based)
      - method (string)
      - data: list-of-rows (strings)
      - html (optional)
      - debug_image (base64 PNG) optional
      - found_header (bool) optional
    """
    summary = {"pages_found": 0, "matches": 0, "methods": {}, "notes": []}
    results_pages: List[Dict[str, Any]] = []
    ext = (filename or "file").lower().split(".")[-1] if filename else ""

    # Patterns for section detection
    patterns = _make_section_patterns(section_keyword)

    # 1) If PDF, attempt text-layer reconstruction first (PyMuPDF/pdfplumber)
    if ext == "pdf":
        # try pdf text-layer tables reconstructed from words (most reliable when present)
        try:
            text_layer_results = _extract_tables_from_pdf_text_layer(file_bytes, section_keyword)
            if text_layer_results:
                for r in text_layer_results:
                    results_pages.append(r)
                    summary["matches"] += 1
                    summary["methods"].setdefault(r.get("method", "pdf_text_reconstruct"), 0)
                    summary["methods"][r.get("method", "pdf_text_reconstruct")] += 1
                summary["pages_found"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}
        except Exception as e:
            logger.exception("pdf text-layer extraction raised: %s", e)
            summary["notes"].append("pdf_text_layer_failed")

    # 2) Rasterize pages (pdf->images) or open image file
    pages_images = []
    # If PDF, try fitz pixmap (prefer) then pdf2image fallback
    if ext == "pdf":
        if fitz is not None:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for pno in range(len(doc)):
                    try:
                        page = doc[pno]
                        pm = page.get_pixmap(dpi=dpi)
                        im = Image.open(BytesIO(pm.tobytes("png"))).convert("RGB")
                        pages_images.append(im)
                    except Exception:
                        # fallback page by page via pdf2image later
                        continue
                try:
                    doc.close()
                except Exception:
                    pass
            except Exception:
                pages_images = []
        if not pages_images and convert_from_bytes is not None:
            try:
                pages_images = convert_from_bytes(file_bytes, dpi=dpi)
            except Exception:
                logger.exception("pdf2image conversion failed")
    else:
        # image file path
        if Image is not None:
            try:
                pages_images = [Image.open(BytesIO(file_bytes)).convert("RGB")]
            except Exception:
                logger.exception("Pillow cannot open image bytes")

    if not pages_images:
        summary["notes"].append("no_pages_images")
        return {"pages": results_pages, "summary": summary}

    # 3) For each page, decide mode and attempt extraction near section keyword if found
    for idx, pil_page in enumerate(pages_images, start=1):
        page_result = {"page": idx, "method": None, "data": [], "html": None}
        # 3a) If this is a PDF (we tried above), use fitz detection to decide page mode
        mode_info = None
        if ext == "pdf" and fitz is not None:
            try:
                # reopen doc to inspect page metadata (fitz)
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                mode_info = _detect_page_mode_with_fitz(doc, idx - 1)
                try:
                    doc.close()
                except Exception:
                    pass
            except Exception:
                mode_info = None
        # if we don't have fitz-based mode info, use simple heuristics: see if PIL image is mostly blank
        if mode_info is None:
            mode_info = {"mode": "unknown", "text_blocks": 0, "text_chars": 0, "image_count": 1}
        mode = mode_info.get("mode", "unknown")
        if debug:
            page_result["debug_mode_info"] = mode_info

        # 3b) OCR search the page for the section keyword (to crop)
        matched_boxes = []
        page_cv = None
        try:
            page_cv = _pil_to_cv(pil_page)
        except Exception:
            page_cv = None
        # Search with OCR text for keyword positions (if pytesseract present)
        if pytesseract is not None and page_cv is not None:
            try:
                odata = pytesseract.image_to_data(page_cv, output_type=Output.DICT, config="--psm 6")
                texts = odata.get("text", [])
                for i, txt in enumerate(texts):
                    if not txt or not txt.strip():
                        continue
                    token = txt.strip()
                    # check against patterns and fuzzy
                    matched = False
                    for pat in patterns:
                        if pat.search(token):
                            matched = True
                            break
                    if not matched and _fuzzy_contains(token, section_keyword, threshold=0.6):
                        matched = True
                    if matched:
                        try:
                            left = int(odata["left"][i]); top = int(odata["top"][i]); width = int(odata["width"][i]); height = int(odata["height"][i])
                            matched_boxes.append(((left, top, width, height), token))
                        except Exception:
                            continue
            except Exception:
                logger.exception("pytesseract search failed on page %s", idx)

        # 3c) If we found a section keyword, crop area and try table extraction inside crop first
        tried_methods = []
        if matched_boxes:
            (x, y, wbox, hbox), matched_txt = matched_boxes[0]
            pad_w = int(pil_page.width * 0.05)
            pad_h = int(pil_page.height * 0.18)
            crop_x0 = max(0, x - pad_w)
            crop_y0 = max(0, y - pad_h)
            crop_x1 = min(pil_page.width, x + wbox + pad_w)
            crop_y1 = min(pil_page.height, y + hbox + pad_h)
            crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            # convert crop to cv
            try:
                crop_cv = _pil_to_cv(crop)
            except Exception:
                crop_cv = None
            # 1) try grid detection in crop
            if crop_cv is not None and cv2 is not None:
                grid_info = _find_table_grid(crop_cv, debug=debug)
                if grid_info.get("cells"):
                    try:
                        rows_text = _ocr_cells_from_boxes(crop_cv, grid_info["cells"])
                        page_result["method"] = "image_grid_cropped"
                        page_result["data"] = rows_text
                        if pd is not None:
                            page_result["html"] = pd.DataFrame(rows_text).to_html(index=False, header=False, border=1)
                        if debug and grid_info.get("overlay") is not None:
                            # overlay is on cropped image; convert to base64
                            page_result["debug_image"] = _img_to_b64png(grid_info["overlay"])
                        tried_methods.append("image_grid_cropped")
                        results_pages.append(page_result)
                        summary["matches"] += 1
                        summary["methods"].setdefault(page_result["method"], 0)
                        summary["methods"][page_result["method"]] += 1
                        continue  # go to next page
                    except Exception:
                        logger.exception("OCR cells extraction failed inside crop")
            # 2) grid not found or failed: attempt OCR cluster in crop
            if crop_cv is not None:
                oc = _ocr_cluster_table_from_image(crop_cv, debug=debug)
                if oc.get("data"):
                    page_result["method"] = "ocr_cluster_cropped"
                    page_result["data"] = oc["data"]
                    if pd is not None:
                        page_result["html"] = pd.DataFrame(oc["data"]).to_html(index=False, header=False, border=1)
                    if debug and oc.get("overlay") is not None:
                        page_result["debug_image"] = _img_to_b64png(oc["overlay"])
                    tried_methods.append("ocr_cluster_cropped")
                    results_pages.append(page_result)
                    summary["matches"] += 1
                    summary["methods"].setdefault(page_result["method"], 0)
                    summary["methods"][page_result["method"]] += 1
                    continue

        # 3d) If no matched keyword or crop attempts failed, try full-page strategies depending on mode
        # Strategy priority:
        #  - If mode == 'vector' try pdfplumber page.extract_tables() if pdfplumber available (we already tried reconstruct above)
        #  - Try grid detection on full page image
        #  - Try OCR clustering fallback (text alignment)
        # Reopen page_cv if needed
        if page_cv is None:
            try:
                page_cv = _pil_to_cv(pil_page)
            except Exception:
                page_cv = None

        # (A) Try pdfplumber structured table (if ext==pdf and pdfplumber available)
        if ext == "pdf" and pdfplumber is not None:
            try:
                # Use pdfplumber to extract tables from the page index
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    if idx - 1 < len(pdf.pages):
                        p = pdf.pages[idx - 1]
                        tables = p.extract_tables()
                        if tables:
                            # choose tables that likely include the section keyword if patterns exist
                            chosen = []
                            for t in tables:
                                flat = " ".join([str(c) for row in t for c in (row or [])]).lower()
                                if patterns:
                                    if any(_fuzzy_contains(flat, k, threshold=0.6) for k in ([section_keyword] + (extra_keywords or []))):
                                        chosen.append(t)
                                else:
                                    chosen.append(t)
                            if not chosen and tables:
                                chosen = tables
                            for t in chosen:
                                data = [[_clean_text(c) if c is not None else "" for c in row] for row in t]
                                page_result = {"page": idx, "method": "pdfplumber_table", "data": data}
                                if pd is not None:
                                    page_result["html"] = pd.DataFrame(data).to_html(index=False, header=False, border=1)
                                results_pages.append(page_result)
                                summary["matches"] += 1
                                summary["methods"].setdefault("pdfplumber_table", 0)
                                summary["methods"]["pdfplumber_table"] += 1
                            if chosen:
                                continue
            except Exception:
                logger.exception("pdfplumber page extraction failed for page %s", idx)

        # (B) Try grid detection on full page image
        if page_cv is not None:
            grid_info = _find_table_grid(page_cv, debug=debug)
            if grid_info.get("cells"):
                try:
                    rows_text = _ocr_cells_from_boxes(page_cv, grid_info["cells"])
                    page_result["method"] = "image_grid_full"
                    page_result["data"] = rows_text
                    if pd is not None:
                        page_result["html"] = pd.DataFrame(rows_text).to_html(index=False, header=False, border=1)
                    if debug and grid_info.get("overlay") is not None:
                        page_result["debug_image"] = _img_to_b64png(grid_info["overlay"])
                    results_pages.append(page_result)
                    summary["matches"] += 1
                    summary["methods"].setdefault(page_result["method"], 0)
                    summary["methods"][page_result["method"]] += 1
                    continue
                except Exception:
                    logger.exception("full-page OCR cells extraction failed")

        # (C) OCR clustering fallback on full page
        if page_cv is not None:
            oc = _ocr_cluster_table_from_image(page_cv, debug=debug)
            if oc.get("data"):
                page_result["method"] = "ocr_cluster_full"
                page_result["data"] = oc["data"]
                if pd is not None:
                    page_result["html"] = pd.DataFrame(oc["data"]).to_html(index=False, header=False, border=1)
                if debug and oc.get("overlay") is not None:
                    page_result["debug_image"] = _img_to_b64png(oc["overlay"])
                results_pages.append(page_result)
                summary["matches"] += 1
                summary["methods"].setdefault(page_result["method"], 0)
                summary["methods"][page_result["method"]] += 1
                continue

        # If no method produced table text, include a note (so user sees page processed but empty)
        page_result["method"] = "none_detected"
        page_result["data"] = []
        results_pages.append(page_result)

    summary["pages_found"] = len(results_pages)
    return {"pages": results_pages, "summary": summary}
