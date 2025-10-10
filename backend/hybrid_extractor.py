# backend/hybrid_extractor.py
"""
Hybrid table extractor for OCR Web Tool.

- Tries PDF text-layer extraction (PyMuPDF/pdfplumber) first.
- If that fails, rasterizes with pdf2image and uses OpenCV + Tesseract OCR with preprocessing.
- Includes tolerant section detection (regex + fuzzy matching).
- Returns structured dict with pages, each page contains data (rows), html, method, debug_image (base64) optionally.
"""

from io import BytesIO
import base64
import logging
import re
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher
from collections import defaultdict

# Defensive imports
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

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
# Small helpers
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
    _, buf = cv2.imencode(".png", img_cv)
    return base64.b64encode(buf).decode("ascii")


def _pil_to_cv(pil_img: "Image.Image"):
    if Image is None or np is None or cv2 is None:
        raise RuntimeError("Pillow/numpy/opencv required for image conversions")
    arr = np.array(pil_img)
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
# Section pattern helpers (FIXED)
# -----------------------
def _make_section_patterns(section_keyword: str) -> List[re.Pattern]:
    """
    Build tolerant regex patterns for a section keyword like "2.14".
    Avoid using re.sub with replacement strings containing backslashes (which caused re.error).
    We construct the pattern by iterating characters of the keyword and escaping non-dots,
    and replacing '.' with a permissive group that allows spaces/dots/commas between the numbers.
    """
    base = (section_keyword or "").strip()
    if not base:
        return []
    # Build fuzzy version safely (no re.sub replacement with backslashes)
    parts = []
    for ch in base:
        if ch == ".":
            # allow flexible separators for dot: spaces, dots, commas repeated
            parts.append(r'[\s\.,]*')
        else:
            parts.append(re.escape(ch))
    base_fuzzy = "".join(parts)

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
            # ignore compilation problems for a variant
            logger.debug("failed to compile pattern variant: %s", v)
    return patterns


# -----------------------
# PDF text-layer table reconstruction
# -----------------------
def _table_from_pdf_text_words(words: List[Tuple[float, float, float, float, str]]) -> List[List[str]]:
    if not words or np is None:
        # simple fallback: return empty
        return []
    # sort by top, then left
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    # group into lines by y tolerance
    tol_y = 6.0
    lines = []
    current = [words_sorted[0]]
    last_y = words_sorted[0][1]
    for w in words_sorted[1:]:
        y = w[1]
        if abs(y - last_y) <= tol_y:
            current.append(w)
            last_y = (last_y + y) / 2.0
        else:
            lines.append(sorted(current, key=lambda x: x[0]))
            current = [w]
            last_y = y
    if current:
        lines.append(sorted(current, key=lambda x: x[0]))

    # build column centers using all words centers
    centers = []
    for ln in lines:
        for (x0, y0, x1, y1, txt) in ln:
            centers.append((x0 + x1) / 2.0)
    if not centers:
        return []

    centers_sorted = sorted(centers)
    gap_thresh = max(20.0, (max(centers_sorted) - min(centers_sorted)) / 30.0)
    col_centers = []
    cluster = [centers_sorted[0]]
    for c in centers_sorted[1:]:
        if abs(c - cluster[-1]) <= gap_thresh:
            cluster.append(c)
        else:
            col_centers.append(sum(cluster) / len(cluster))
            cluster = [c]
    col_centers.append(sum(cluster) / len(cluster))

    # merge very close centers
    merged = []
    for c in col_centers:
        if not merged:
            merged.append(c)
        else:
            if abs(c - merged[-1]) < gap_thresh / 2:
                merged[-1] = (merged[-1] + c) / 2
            else:
                merged.append(c)
    col_centers = merged

    # compose rows
    table = []
    for ln in lines:
        row = [""] * len(col_centers)
        for (x0, y0, x1, y1, txt) in ln:
            cx = (x0 + x1) / 2.0
            # nearest center
            dists = [abs(cx - cc) for cc in col_centers]
            if len(dists) == 0:
                col_idx = 0
            else:
                col_idx = int(min(range(len(dists)), key=lambda i: dists[i]))
            if row[col_idx]:
                row[col_idx] += " " + txt
            else:
                row[col_idx] = txt
        # keep only rows with some content
        if any(c.strip() for c in row):
            table.append([_clean_text(c) for c in row])
    return table


def _extract_table_from_pdf_text_layer(pdf_bytes: bytes, section_keyword: str, extra_keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    page_results = []
    patterns = _make_section_patterns(section_keyword)
    try:
        if fitz is not None:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for i, page in enumerate(doc, start=1):
                try:
                    words = page.get_text("words")  # (x0,y0,x1,y1,word,...)
                except Exception:
                    words = []
                if not words:
                    continue
                page_text = " ".join([w[4] for w in words]).lower()
                found_header = any(p.search(page_text) for p in patterns)
                # build tuples
                words_tuples = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
                table = _table_from_pdf_text_words(words_tuples)
                # heuristics: table-like if multiple rows and columns
                if table and len(table) >= 2 and any(len(r) > 1 for r in table):
                    df_html = pd.DataFrame(table).to_html(index=False, header=False, border=1) if pd is not None else None
                    page_results.append({
                        "page": i,
                        "method": "pdf_text_table_via_fitz",
                        "data": table,
                        "html": df_html,
                        "found_header": found_header,
                    })
            try:
                doc.close()
            except Exception:
                pass
            if page_results:
                return page_results
    except Exception as e:
        logger.exception("PyMuPDF text-layer attempt failed: %s", e)

    # fallback to pdfplumber words if available
    try:
        if pdfplumber is not None:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        words = page.extract_words(use_text_flow=True)
                    except Exception:
                        words = []
                    if not words:
                        continue
                    page_text = " ".join([w.get("text", "") for w in words]).lower()
                    words_tuples = []
                    for w in words:
                        try:
                            x0 = float(w.get("x0", 0)); top = float(w.get("top", 0)); x1 = float(w.get("x1", 0)); bottom = float(w.get("bottom", 0))
                            txt = w.get("text", "")
                            words_tuples.append((x0, top, x1, bottom, txt))
                        except Exception:
                            continue
                    table = _table_from_pdf_text_words(words_tuples)
                    if table and len(table) >= 2 and any(len(r) > 1 for r in table):
                        df_html = pd.DataFrame(table).to_html(index=False, header=False, border=1) if pd is not None else None
                        page_results.append({
                            "page": i,
                            "method": "pdf_text_table_via_pdfplumber",
                            "data": table,
                            "html": df_html,
                            "found_header": any(p.search(page_text) for p in patterns),
                        })
            if page_results:
                return page_results
    except Exception as e:
        logger.exception("pdfplumber text-layer attempt failed: %s", e)

    return []


# -----------------------
# Image preprocessing
# -----------------------
def _preprocess_for_table(cv_img: "np.ndarray", max_dim: int = 2000) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    if cv_img is None or cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy required")
    h, w = cv_img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 9)
    return bin_img, gray, cv_img


# -----------------------
# Extract table by image (lines or clustering)
# -----------------------
def _extract_table_from_image_region(pil_img: "Image.Image", debug: bool = False) -> Dict[str, Any]:
    result = {"method": "ocr_fallback", "data": [], "html": "", "debug_image": None, "rows": 0, "cols": 0}
    if Image is None or cv2 is None or pytesseract is None or np is None:
        result["summary"] = {"error": "missing Pillow/opencv/pytesseract/numpy"}
        return result

    orig_cv = _pil_to_cv(pil_img)
    bin_img, gray, orig_img = _preprocess_for_table(orig_cv)

    h, w = bin_img.shape[:2]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 20), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    horiz_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    mask = cv2.add(horiz_lines, vert_lines)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww * hh < 800:
            continue
        boxes.append((x, y, ww, hh))

    # If boxes look like cells -> build grid
    if boxes and len(boxes) >= 4:
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        rows = []
        current_row = [boxes[0]]
        last_y = boxes[0][1]
        for b in boxes[1:]:
            if abs(b[1] - last_y) <= max(10, int(b[3] * 0.5)):
                current_row.append(b)
                last_y = int((last_y + b[1]) / 2)
            else:
                rows.append(sorted(current_row, key=lambda r: r[0]))
                current_row = [b]
                last_y = b[1]
        if current_row:
            rows.append(sorted(current_row, key=lambda r: r[0]))

        col_centers = []
        if rows:
            largest_row = max(rows, key=lambda r: len(r))
            centers = [int(x + wbox / 2) for (x, y, wbox, hbox) in largest_row]
            col_centers = sorted(centers)

        grid = []
        for r in rows:
            row_cells = [""] * max(1, len(col_centers))
            for (x, y, ww, hh) in r:
                cx = int(x + ww / 2)
                if not col_centers:
                    col_idx = 0
                else:
                    col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
                ex, ey = max(0, x - 2), max(0, y - 2)
                ew, eh = min(ww + 4, orig_img.shape[1] - ex), min(hh + 4, orig_img.shape[0] - ey)
                crop = orig_img[ey:ey + eh, ex:ex + ew]
                try:
                    data = pytesseract.image_to_string(crop, config="--psm 6")
                except Exception:
                    data = ""
                txt = _clean_text(data)
                if row_cells[col_idx]:
                    row_cells[col_idx] += " " + txt if txt else ""
                else:
                    row_cells[col_idx] = txt
            grid.append([_clean_text(c) for c in row_cells])

        df = pd.DataFrame(grid).replace(r'^\s*$', "", regex=True) if pd is not None else None
        result["method"] = "image_table_lines"
        result["data"] = df.to_numpy().astype(str).tolist() if df is not None else grid
        result["html"] = df.to_html(index=False, header=False, border=1) if df is not None else ""
        result["rows"], result["cols"] = (df.shape if df is not None else (len(grid), max((len(r) for r in grid), default=0)))
        if debug:
            overlay = orig_img.copy()
            for (x, y, ww, hh) in boxes:
                cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 128, 255), 2)
            if col_centers:
                for c in col_centers:
                    cv2.line(overlay, (int(c), 0), (int(c), overlay.shape[0]), (0, 200, 0), 1)
            result["debug_image"] = _img_to_b64png(overlay)
        return result

    # Fallback: OCR word clustering
    try:
        odata = pytesseract.image_to_data(orig_img, output_type=Output.DICT, config="--psm 6")
    except Exception:
        logger.exception("Tesseract image_to_data failed")
        try:
            text = pytesseract.image_to_string(orig_img, config="--psm 6")
            rows = [r.strip() for r in text.splitlines() if r.strip()]
            result["method"] = "ocr_fallback_simple"
            result["data"] = [[r] for r in rows]
            result["html"] = pd.DataFrame(result["data"]).to_html(index=False, header=False, border=1) if pd is not None else ""
            result["rows"], result["cols"] = len(result["data"]), 1
            return result
        except Exception as e:
            result["summary"] = {"error": f"full OCR fallback failed: {e}"}
            return result

    words = []
    n = len(odata.get("text", []))
    for i in range(n):
        t = (odata["text"][i] or "").strip()
        if not t:
            continue
        try:
            left = int(odata.get("left", [0])[i])
            top = int(odata.get("top", [0])[i])
            width = int(odata.get("width", [0])[i])
            height = int(odata.get("height", [0])[i])
        except Exception:
            continue
        conf = int(odata.get("conf", [-1])[i]) if odata.get("conf") else -1
        words.append({"text": t, "left": left, "top": top, "width": width, "height": height, "conf": conf})

    if not words:
        result["summary"] = {"error": "no OCR words"}
        return result

    words_sorted = sorted(words, key=lambda w: (w["top"], w["left"]))
    rows_clusters = []
    current_row = []
    last_y = None
    for w in words_sorted:
        if last_y is None:
            current_row = [w]
            last_y = w["top"]
        else:
            if abs(w["top"] - last_y) <= max(10, int(w["height"] * 0.6)):
                current_row.append(w)
                last_y = int(sum([x["top"] for x in current_row]) / len(current_row))
            else:
                rows_clusters.append(sorted(current_row, key=lambda x: x["left"]))
                current_row = [w]
                last_y = w["top"]
    if current_row:
        rows_clusters.append(sorted(current_row, key=lambda x: x["left"]))

    centers = []
    for r in rows_clusters:
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

    grid = []
    for r in rows_clusters:
        row_cells = ["" for _ in range(len(col_centers))]
        for w in r:
            cx = int(w["left"] + w["width"] / 2.0)
            col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + w["text"]
            else:
                row_cells[col_idx] = w["text"]
        grid.append([_clean_text(c) for c in row_cells])

    maxcols = max(len(r) for r in grid) if grid else len(col_centers)
    for r in grid:
        while len(r) < maxcols:
            r.append("")

    df = pd.DataFrame(grid).replace(r'^\s*$', "", regex=True) if pd is not None else None
    result["method"] = "image_table_ocr_cluster"
    result["data"] = df.to_numpy().astype(str).tolist() if df is not None else grid
    result["html"] = df.to_html(index=False, header=False, border=1) if df is not None else ""
    result["rows"], result["cols"] = (df.shape if df is not None else (len(grid), max((len(r) for r in grid), default=0)))
    if debug:
        overlay = orig_img.copy()
        for w in words_sorted:
            cv2.rectangle(overlay, (w["left"], w["top"]), (w["left"] + w["width"], w["top"] + w["height"]), (255, 0, 0), 1)
        for c in col_centers:
            cv2.line(overlay, (int(c), 0), (int(c), overlay.shape[0]), (0, 255, 0), 1)
        result["debug_image"] = _img_to_b64png(overlay)
    return result


# -----------------------
# Public extract_tables entry point
# -----------------------
def extract_tables(file_bytes: bytes,
                   filename: str,
                   section_keyword: str = "2.14",
                   extra_keywords: Optional[List[str]] = None,
                   debug: bool = False,
                   dpi: int = 300) -> Dict[str, Any]:
    keywords = [section_keyword] if section_keyword else []
    if extra_keywords:
        keywords += extra_keywords
    logger.info("extract_tables called filename=%s section=%s dpi=%s debug=%s", filename, section_keyword, dpi, debug)

    results_pages: List[Dict[str, Any]] = []
    summary = {"pages_found": 0, "matches": 0, "methods": {}}
    fname = (filename or "file").lower()
    ext = fname.split(".")[-1] if "." in fname else ""

    # 1) Try PDF text-layer reconstruction (most reliable for vector AIPs)
    if ext == "pdf":
        try:
            page_results = _extract_table_from_pdf_text_layer(file_bytes, section_keyword, extra_keywords)
            if page_results:
                for pr in page_results:
                    results_pages.append(pr)
                    summary["matches"] += 1
                    summary["methods"].setdefault(pr.get("method", "pdf_text"), 0)
                    summary["methods"][pr.get("method", "pdf_text")] += 1
                summary["pages_found"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}
        except Exception as e:
            logger.exception("pdf text-layer attempt raised: %s", e)

    # 2) Rasterize pages (pdf -> images) or open image
    pages_images = []
    if ext == "pdf":
        if convert_from_bytes is None or Image is None:
            logger.warning("pdf2image or Pillow missing; cannot rasterize pdf")
        else:
            try:
                pages_images = convert_from_bytes(file_bytes, dpi=dpi)
            except Exception:
                logger.exception("pdf2image conversion failed")
                pages_images = []
    else:
        if Image is None:
            logger.warning("Pillow missing; cannot open image")
        else:
            try:
                pages_images = [Image.open(BytesIO(file_bytes)).convert("RGB")]
            except Exception:
                logger.exception("Pillow cannot open image; not an image?")

    # 3) Try OCR-based search for section on each page and crop region around it
    if pages_images:
        patterns = _make_section_patterns(section_keyword)
        for idx, pil_page in enumerate(pages_images, start=1):
            # OCR search of full page to find keyword positions
            matched_boxes = []
            if pytesseract is not None and cv2 is not None:
                try:
                    cv_img = _pil_to_cv(pil_page)
                    odata = pytesseract.image_to_data(cv_img, output_type=Output.DICT, config="--psm 6")
                    texts = odata.get("text", [])
                    for i, txt in enumerate(texts):
                        if not txt or not txt.strip():
                            continue
                        t = txt.strip()
                        for pat in patterns:
                            if pat.search(t) or _fuzzy_contains(t, section_keyword, threshold=0.6):
                                try:
                                    x = int(odata["left"][i]); y = int(odata["top"][i]); w = int(odata["width"][i]); h = int(odata["height"][i])
                                    matched_boxes.append(((x, y, w, h), t))
                                except Exception:
                                    continue
                except Exception:
                    logger.exception("pytesseract page search failed")
            # If found, crop near first match and try table extraction
            if matched_boxes:
                (x, y, wbox, hbox), matched_txt = matched_boxes[0]
                pad_w = int(pil_page.width * 0.05)
                pad_h = int(pil_page.height * 0.18)
                crop_x0 = max(0, x - pad_w)
                crop_y0 = max(0, y - pad_h)
                crop_x1 = min(pil_page.width, x + wbox + pad_w)
                crop_y1 = min(pil_page.height, y + hbox + pad_h)
                crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                out = _extract_table_from_image_region(crop, debug=debug)
                out.update({"page": idx, "method_origin": "ocr_page_search", "matched_keyword": matched_txt, "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
                results_pages.append(out)
                summary["matches"] += 1
                summary["methods"].setdefault(out["method"], 0)
                summary["methods"][out["method"]] += 1

        if results_pages:
            summary["pages_found"] = len(results_pages)
            return {"pages": results_pages, "summary": summary}

    # 4) Try pdfplumber structured table extraction (if installed)
    if ext == "pdf" and pdfplumber is not None:
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        tables = []
                    for t in tables:
                        if not t:
                            continue
                        # check if any cell contains the section keyword (fuzzy)
                        flat = " ".join([str(c) for row in t for c in (row or [])]).lower()
                        if any(_fuzzy_contains(flat, k, threshold=0.6) for k in (keywords if keywords else [section_keyword])):
                            df = pd.DataFrame(t).fillna("").applymap(_clean_text) if pd is not None else None
                            out = {"page": i, "method": "pdfplumber_table", "data": df.to_numpy().astype(str).tolist() if df is not None else t, "html": df.to_html(index=False, header=False, border=1) if df is not None else None}
                            results_pages.append(out)
                            summary["matches"] += 1
                            summary["methods"].setdefault("pdfplumber_table", 0)
                            summary["methods"]["pdfplumber_table"] += 1
            if results_pages:
                summary["pages_found"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}
        except Exception:
            logger.exception("pdfplumber attempt failed")

    # 5) Final fallback: attempt full-page table extraction on first page image
    if pages_images:
        pil_page = pages_images[0]
        out = _extract_table_from_image_region(pil_page, debug=debug)
        out.update({"page": 1, "method_origin": "final_full_page_attempt"})
        results_pages.append(out)
        summary["methods"].setdefault(out["method"], 0)
        summary["methods"][out["method"]] += 1
        summary["pages_found"] = len(results_pages)
        return {"pages": results_pages, "summary": summary}

    # nothing processed
    return {"pages": results_pages, "summary": {"error": "no pages/images available or required libs missing"}}
