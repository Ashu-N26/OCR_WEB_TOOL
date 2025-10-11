# backend/hybrid_extractor.py
"""
Hybrid extractor (complete end-to-end)

Usage:
    from backend.hybrid_extractor import extract_tables
    result = extract_tables(file_bytes, filename="EDTM_202536_09.pdf", section_keyword="2.14", debug=True)

Returns:
    {
      "pages": [
         {
           "page": <1-based page number>,
           "method": <string>,
           "data": <list of rows (list of strings)> OR <list of lines>,
           "html": <optional HTML table>,
           "debug_image": <base64 PNG overlay optional>,
           "found_header": <bool optional>
         }, ...
      ],
      "summary": {
         "pages_found": n,
         "matches": m,
         "methods": {...},
         "notes": [...]
      }
    }
"""

import io
import re
import base64
import logging
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher

# Optional deps - import defensively
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
    from PIL import Image, ImageFilter, ImageEnhance
except Exception:
    Image = None
    ImageFilter = None
    ImageEnhance = None

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
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------
# Helper utilities
# -----------------------
def _clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x0c", " ").replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s


def _img_to_b64png(img_cv) -> str:
    if cv2 is None:
        return ""
    try:
        _, buf = cv2.imencode(".png", img_cv)
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return ""


def _pil_to_cv(pil_img):
    if Image is None or np is None or cv2 is None:
        raise RuntimeError("Pillow/numpy/opencv required for image conversions")
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    if not haystack or not needle:
        return False
    hay = haystack.lower()
    key = needle.lower()
    if key in hay:
        return True
    # token window fuzzy match
    tokens = hay.split()
    n = len(tokens)
    m = len(key.split())
    if m == 0:
        return False
    for window in range(m, min(n, m + 6) + 1):
        for i in range(0, n - window + 1):
            sub = " ".join(tokens[i:i + window])
            ratio = SequenceMatcher(None, sub, key).ratio()
            if ratio >= threshold:
                return True
    if SequenceMatcher(None, hay, key).ratio() >= threshold:
        return True
    return False


# -----------------------
# Safe section pattern builder (avoid re.sub replacement issues)
# -----------------------
def _make_section_patterns(section_keyword: Optional[str]) -> List[re.Pattern]:
    base = (section_keyword or "").strip()
    if not base:
        return []
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
        rf"{base_fuzzy}.*ilumin",
        rf"aproxima[cç][aã]o.*{base_fuzzy}",
    ]
    patterns = []
    for v in variants:
        try:
            patterns.append(re.compile(v, re.IGNORECASE | re.DOTALL))
        except Exception:
            continue
    return patterns


# -----------------------
# Post-processor integrated (advanced)
# -----------------------
# This is adapted and condensed from a robust implementation:
from collections import Counter


def _clean_cell(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x0c", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    s = re.sub(r"^[\s\-\u2013\u2014\.:,]+", "", s)
    s = re.sub(r"[\s\-\u2013\u2014\.:,]+$", "", s)
    return s


def _rectify_table(rows: List[List[str]]) -> List[List[str]]:
    max_cols = 0
    for r in rows:
        if r is None:
            continue
        max_cols = max(max_cols, len(r))
    if max_cols == 0:
        return []
    out = []
    for r in rows:
        r = r or []
        new = [(_clean_cell(c) if c is not None else "") for c in r]
        if len(new) < max_cols:
            new.extend([""] * (max_cols - len(new)))
        out.append(new[:max_cols])
    return out


def _row_fingerprint(row: List[str]) -> str:
    t = " ".join([c.lower() for c in row if c and len(c.strip()) > 0])
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    t = " ".join(t.split())
    return t


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _most_common_nonzero_counts(rows: List[List[str]]) -> int:
    counts = [sum(1 for c in r if c.strip()) for r in rows]
    counts = [c for c in counts if c > 0]
    if not counts:
        return 0
    ctr = Counter(counts)
    most, _ = ctr.most_common(1)[0]
    return most


def _split_cell_on_multispace(cell: str) -> List[str]:
    if not cell or not cell.strip():
        return [cell]
    parts = re.split(r"\s{2,}", cell)
    parts = [p.strip() for p in parts if p is not None and p.strip()]
    return parts if parts else [cell.strip()]


def _distribute_splits_to_row(row: List[str], target_cols: int, debug: bool = False) -> Tuple[List[str], bool]:
    nonempty = [i for i, c in enumerate(row) if c.strip()]
    if len(nonempty) >= target_cols:
        return row, False
    for idx in nonempty:
        parts = _split_cell_on_multispace(row[idx])
        if len(parts) > 1:
            new = row[:idx] + parts + row[idx + 1:]
            if len(new) < target_cols:
                new.extend([""] * (target_cols - len(new)))
            if len(new) > target_cols:
                extra = new[target_cols - 1:]
                new = new[:target_cols]
                new[target_cols - 1] = (new[target_cols - 1] + " " + " ".join(extra)).strip()
            if debug:
                logger.debug(f"_distribute_splits_to_row: split cell {idx} into {len(parts)} parts")
            return new, True
    return row, False


def _merge_continuation_rows(rows: List[List[str]], debug: bool = False) -> Tuple[List[List[str]], int]:
    i = 0
    merges = 0
    out = []
    while i < len(rows):
        cur = rows[i]
        if i + 1 >= len(rows):
            out.append(cur)
            break
        nxt = rows[i + 1]
        cur_non = sum(1 for c in cur if c.strip())
        nxt_non = sum(1 for c in nxt if c.strip())
        nxt_len = sum(len(c) for c in nxt if c and c.strip())
        if nxt_non > 0 and nxt_non <= 2 and nxt_len <= 80:
            merged = [c for c in cur]
            for j, cell in enumerate(nxt):
                if not cell.strip():
                    continue
                target_idx = None
                if cur_non > 0:
                    if j < len(merged) and not merged[j].strip():
                        target_idx = j
                    else:
                        left = None
                        for k in range(min(j, len(merged) - 1), -1, -1):
                            if merged[k].strip():
                                left = k
                                break
                        target_idx = left if left is not None else j
                else:
                    target_idx = j
                if target_idx is None or target_idx >= len(merged):
                    target_idx = min(len(merged) - 1, j)
                if merged[target_idx].strip():
                    merged[target_idx] = merged[target_idx] + " " + cell.strip()
                else:
                    merged[target_idx] = cell.strip()
            out.append([_clean_cell(x) for x in merged])
            merges += 1
            if debug:
                logger.debug(f"Merged row {i+1} into row {i} (0-based {i})")
            i += 2
        else:
            out.append(cur)
            i += 1
    return out, merges


def _remove_repeated_headers(rows: List[List[str]], debug: bool = False) -> Tuple[List[List[str]], List[int]]:
    removed = []
    if not rows:
        return rows, removed
    top_region = rows[:3]
    counts = [sum(1 for c in r if c.strip()) for r in top_region]
    if not counts or max(counts) == 0:
        return rows, removed
    header_idx_local = max(range(len(top_region)), key=lambda i: counts[i])
    header_row = rows[header_idx_local]
    header_fp = _row_fingerprint(header_row)
    new = []
    for i, r in enumerate(rows):
        fp = _row_fingerprint(r)
        if i != header_idx_local and header_fp and fp:
            if _similar(header_fp, fp) > 0.85:
                removed.append(i)
                if debug:
                    logger.debug(f"Removed repeated header at row {i}, similarity={_similar(header_fp, fp):.2f}")
                continue
        new.append(r)
    return new, removed


def _normalize_and_finalize(rows: List[List[str]]) -> List[List[str]]:
    maxc = max((len(r) for r in rows), default=0)
    out = []
    for r in rows:
        new = [(_clean_cell(c) if c is not None else "") for c in r]
        if len(new) < maxc:
            new.extend([""] * (maxc - len(new)))
        out.append(new[:maxc])
    return out


def postprocess_table(table_rows: List[List[str]], debug: bool = False, expected_cols: Optional[int] = None) -> Dict[str, Any]:
    notes = []
    if not table_rows:
        return {"table": [], "format": "empty", "removed_header_indices": [], "merged_rows": 0, "notes": ["empty_input"]}
    rect = _rectify_table(table_rows)
    if debug:
        notes.append(f"rectified_rows={len(rect)}, cols={len(rect[0]) if rect else 0}")
    if expected_cols and isinstance(expected_cols, int) and expected_cols > 0:
        target_cols = expected_cols
    else:
        target_cols = _most_common_nonzero_counts(rect)
        if target_cols <= 1:
            lines = [r[0] if r else "" for r in rect]
            return {"table": [[_clean_cell(l)] for l in lines], "format": "lines", "removed_header_indices": [], "merged_rows": 0, "notes": notes}
    notes.append(f"initial_target_cols={target_cols}")
    changed = True
    attempts = 0
    while changed and attempts < 5:
        changed = False
        attempts += 1
        for i, row in enumerate(rect):
            nonempty = sum(1 for c in row if c.strip())
            if nonempty < target_cols:
                new_row, did = _distribute_splits_to_row(row, target_cols, debug=debug)
                if did:
                    rect[i] = new_row
                    changed = True
        if debug:
            notes.append(f"redistribute_attempt={attempts}, changed={changed}")
    rect, merges = _merge_continuation_rows(rect, debug=debug)
    notes.append(f"merged_rows={merges}")
    changed = True
    attempts = 0
    while changed and attempts < 3:
        changed = False
        attempts += 1
        for i, row in enumerate(rect):
            nonempty = sum(1 for c in row if c.strip())
            if nonempty < target_cols:
                new_row, did = _distribute_splits_to_row(row, target_cols, debug=debug)
                if did:
                    rect[i] = new_row
                    changed = True
        if debug:
            notes.append(f"postmerge_redistribute_attempt={attempts}, changed={changed}")
    rect, removed = _remove_repeated_headers(rect, debug=debug)
    notes.append(f"removed_headers_count={len(removed)}")
    final = _normalize_and_finalize(rect)
    return {"table": final, "format": "table", "removed_header_indices": removed, "merged_rows": merges, "notes": notes}


# -----------------------
# PDF text-layer reconstruction (if textual)
# -----------------------
def _table_from_words(words: List[Tuple[float, float, float, float, str]]) -> List[List[str]]:
    if not words or np is None:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
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
    table = []
    for ln in lines:
        row = ["" for _ in col_centers]
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


def _extract_tables_from_pdf_text_layer(pdf_bytes: bytes, section_patterns: List[re.Pattern]) -> List[Dict[str, Any]]:
    results = []
    if fitz is None:
        return results
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return results
    try:
        for i in range(len(doc)):
            try:
                page = doc[i]
                words = page.get_text("words") or []
                if not words:
                    continue
                page_text = " ".join([w[4] for w in words if len(w) >= 5])
                found_header = any(p.search(page_text) for p in section_patterns) if section_patterns else False
                words_tuples = []
                for w in words:
                    try:
                        if len(w) >= 5:
                            x0, y0, x1, y1, txt = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
                            words_tuples.append((x0, y0, x1, y1, txt))
                    except Exception:
                        continue
                table = _table_from_words(words_tuples)
                if table and len(table) >= 2 and any(len(r) > 1 for r in table):
                    df_html = pd.DataFrame(table).to_html(index=False, header=False, border=1) if pd is not None else None
                    results.append({"page": i + 1, "method": "pdf_text_reconstruct", "data": table, "html": df_html, "found_header": found_header})
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return results


# -----------------------
# Image preprocessing and grid detection
# -----------------------
def _preprocess_for_table(cv_img, max_dim: int = 2400):
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy required")
    h, w = cv_img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    return bin_img, gray, cv_img


def _find_table_grid(cv_img, debug: bool = False) -> Dict[str, Any]:
    result = {"cells": [], "overlay": None}
    if cv2 is None or np is None:
        return result
    try:
        bin_img, gray, orig = _preprocess_for_table(cv_img)
        h, w = bin_img.shape
        horiz_size = max(10, w // 30)
        vert_size = max(10, h // 40)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        horiz = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        mask = cv2.add(horiz, vert)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if ww * hh < 800:
                continue
            boxes.append((x, y, ww, hh))
        if not boxes:
            # fallback: try HoughLinesP if needed (skipped complexity)
            result["cells"] = []
            return result
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
        col_centers = []
        if rows:
            largest_row = max(rows, key=lambda r: len(r))
            centers = [int(x + wbox / 2) for (x, y, wbox, hbox) in largest_row]
            col_centers = sorted(centers)
        grid = []
        for r in rows:
            row_cells = [None] * max(1, len(col_centers))
            for (x, y, ww, hh) in r:
                cx = int(x + ww / 2)
                if not col_centers:
                    col_idx = 0
                else:
                    col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
                if col_idx >= len(row_cells):
                    continue
                row_cells[col_idx] = (x, y, ww, hh)
            grid.append(row_cells)
        result["cells"] = grid
        if debug:
            overlay = orig.copy()
            for row in grid:
                for box in row:
                    if box is None:
                        continue
                    x, y, ww, hh = box
                    cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 180, 255), 2)
            result["overlay"] = overlay
    except Exception as e:
        logger.exception("find_table_grid failed: %s", e)
    return result


def _ocr_cells_from_boxes(orig_img, cell_boxes: List[List[Optional[Tuple[int, int, int, int]]]]) -> List[List[str]]:
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


def _ocr_cluster_table_from_image(orig_img, debug: bool = False) -> Dict[str, Any]:
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
# Main public entrypoint
# -----------------------
def extract_tables(file_bytes: bytes,
                   filename: str,
                   section_keyword: Optional[str] = "2.14",
                   extra_keywords: Optional[List[str]] = None,
                   debug: bool = False,
                   dpi: int = 300) -> Dict[str, Any]:
    """
    file_bytes: bytes of uploaded file
    filename: original filename (used to detect extension)
    section_keyword: e.g., "2.14" or "2.14 approach"
    extra_keywords: list of other keywords to prefer
    debug: if True include overlays and notes
    dpi: rasterization DPI for scanned PDFs
    """
    summary = {"pages_found": 0, "matches": 0, "methods": {}, "notes": []}
    results_pages: List[Dict[str, Any]] = []
    ext = (filename or "file").lower().split(".")[-1] if filename else ""

    patterns = _make_section_patterns(section_keyword) if section_keyword else []

    # 1) Try text-layer reconstruction for PDF (fast & precise when available)
    if ext == "pdf":
        try:
            text_layer_results = _extract_tables_from_pdf_text_layer(file_bytes, patterns)
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

    # 2) Rasterize pages (PDF->images) or open image file
    pages_images = []
    if ext == "pdf":
        # Try fitz first
        if fitz is not None:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for pno in range(len(doc)):
                    try:
                        page = doc[pno]
                        pm = page.get_pixmap(dpi=dpi, alpha=False)
                        im = Image.open(io.BytesIO(pm.tobytes("png"))).convert("RGB")
                        pages_images.append(im)
                    except Exception:
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
        if Image is not None:
            try:
                pages_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
            except Exception:
                logger.exception("Pillow cannot open image bytes")

    if not pages_images:
        summary["notes"].append("no_pages_images")
        return {"pages": results_pages, "summary": summary}

    # 3) For each page: search for the section keyword via OCR and attempt table extraction
    for idx, pil_page in enumerate(pages_images, start=1):
        page_result = {"page": idx, "method": None, "data": [], "html": None}
        try:
            page_cv = None
            try:
                page_cv = _pil_to_cv(pil_page)
            except Exception:
                page_cv = None

            # 3a) OCR-scan page for section keyword (if patterns available)
            matched_boxes = []
            page_text_full = ""
            if pytesseract is not None and page_cv is not None:
                try:
                    odata = pytesseract.image_to_data(page_cv, output_type=Output.DICT, config="--psm 6")
                    texts = odata.get("text", [])
                    page_text_full = " ".join([t for t in texts if t and t.strip()])
                    for i, txt in enumerate(texts):
                        if not txt or not txt.strip():
                            continue
                        token = txt.strip()
                        matched = False
                        for pat in patterns:
                            if pat.search(token):
                                matched = True
                                break
                        if not matched and _fuzzy_contains(token, section_keyword or "", threshold=0.6):
                            matched = True
                        if matched:
                            try:
                                left = int(odata["left"][i]); top = int(odata["top"][i]); width = int(odata["width"][i]); height = int(odata["height"][i])
                                matched_boxes.append(((left, top, width, height), token))
                            except Exception:
                                continue
                except Exception:
                    logger.exception("pytesseract search failed on page %s", idx)
            else:
                # fallback to basic text for section checks
                if fitz is not None and ext == "pdf":
                    try:
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        page_text_full = doc[idx - 1].get_text("text") or ""
                        try:
                            doc.close()
                        except Exception:
                            pass
                    except Exception:
                        page_text_full = ""
            # If OCR page_text doesn't contain section_keyword, but page_text_full fuzzy contains, mark matched
            page_contains_section = False
            if page_text_full:
                for pat in patterns:
                    if pat.search(page_text_full):
                        page_contains_section = True
                        break
                if not page_contains_section and _fuzzy_contains(page_text_full, section_keyword or "", threshold=0.55):
                    page_contains_section = True

            # 3b) If found matched_boxes, crop area and try table extraction in crop
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
                try:
                    crop_cv = _pil_to_cv(crop)
                except Exception:
                    crop_cv = None
                if crop_cv is not None:
                    # try grid
                    grid_info = _find_table_grid(crop_cv, debug=debug)
                    if grid_info.get("cells"):
                        try:
                            rows_text = _ocr_cells_from_boxes(crop_cv, grid_info["cells"])
                            page_result["method"] = "image_grid_cropped"
                            page_result["data"] = rows_text
                            if pd is not None:
                                page_result["html"] = pd.DataFrame(rows_text).to_html(index=False, header=False, border=1)
                            if debug and grid_info.get("overlay") is not None:
                                page_result["debug_image"] = _img_to_b64png(grid_info["overlay"])
                            tried_methods.append("image_grid_cropped")
                            results_pages.append(page_result)
                            summary["matches"] += 1
                            summary["methods"].setdefault(page_result["method"], 0)
                            summary["methods"][page_result["method"]] += 1
                            continue
                        except Exception:
                            logger.exception("OCR cells extraction failed inside crop")
                    # cluster fallback on crop
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

            # 3c) If no matched keyword OR crop attempts failed: try full page strategies
            # A: pdfplumber tables (if available & pdf)
            if ext == "pdf" and pdfplumber is not None:
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        if idx - 1 < len(pdf.pages):
                            p = pdf.pages[idx - 1]
                            tables = p.extract_tables()
                            if tables:
                                chosen = []
                                for t in tables:
                                    flat = " ".join([str(c) for row in t for c in (row or [])]).lower()
                                    matches_section = False
                                    if patterns:
                                        for key_pat in patterns:
                                            if key_pat.search(flat) or (_fuzzy_contains(flat, section_keyword or "", threshold=0.6)):
                                                matches_section = True
                                                break
                                    if not patterns or matches_section:
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

            # B: full-page grid detection
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

            # C: OCR clustering fallback on full page
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

            # If no table detected, attach page_result with method none_detected
            page_result["method"] = "none_detected"
            page_result["data"] = []
            results_pages.append(page_result)

        except Exception as e:
            logger.exception("Exception processing page %s: %s", idx, e)
            results_pages.append({"page": idx, "method": "error", "data": [], "error": str(e)})

    # 4) If nothing matched (no tables), apply a fallback: full-document OCR + postprocess (to avoid "no tables detected")
    if summary["matches"] == 0:
        summary["notes"].append("no_tables_found_fallback_fullscan")
        fallback_rows = []
        for idx, pil_page in enumerate(pages_images, start=1):
            try:
                page_cv = _pil_to_cv(pil_page) if pil_page is not None and Image is not None else None
                if page_cv is None or pytesseract is None:
                    continue
                odata = pytesseract.image_to_data(page_cv, output_type=Output.DICT, config="--psm 6")
                words = []
                n = len(odata.get("text", []))
                for i in range(n):
                    txt = (odata["text"][i] or "").strip()
                    if not txt:
                        continue
                    left = int(odata.get("left", [0])[i])
                    top = int(odata.get("top", [0])[i])
                    words.append((left, top, txt))
                # cluster into rows
                words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
                cur_row = []
                last_y = None
                for w in words_sorted:
                    if last_y is None:
                        last_y = w[1]
                    if abs(w[1] - last_y) > 12:
                        fallback_rows.append([" ".join([t for (_, _, t) in cur_row])])
                        cur_row = []
                    cur_row.append(w)
                    last_y = w[1]
                if cur_row:
                    fallback_rows.append([" ".join([t for (_, _, t) in cur_row])])
            except Exception:
                continue
        # run postprocessor on fallback rows (line mode)
        if fallback_rows:
            pp = postprocess_table(fallback_rows, debug=debug)
            # return as a single page result
            result_page = {"page": 1, "method": "fallback_full_ocr", "data": pp.get("table") if pp else [], "html": None}
            if pd is not None and pp.get("table"):
                try:
                    result_page["html"] = pd.DataFrame(pp.get("table")).to_html(index=False, header=False, border=1)
                except Exception:
                    result_page["html"] = None
            results_pages = [result_page]
            summary["matches"] = 1
            summary["methods"].setdefault("fallback_full_ocr", 0)
            summary["methods"]["fallback_full_ocr"] += 1

    summary["pages_found"] = len(results_pages)
    return {"pages": results_pages, "summary": summary}

