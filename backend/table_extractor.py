# backend/table_extractor.py
import io
import tempfile
import os
from typing import List, Dict, Any

from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd

# Try importing Camelot; optional (used for digital PDFs)
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

from pdf2image import convert_from_bytes


def _camelot_extract(pdf_path: str) -> List[Dict[str, Any]]:
    """Try extracting tables via Camelot (works for digital PDFs with clear table borders)."""
    results = []
    if not _HAS_CAMELOT:
        return results

    try:
        # Try lattice first (grid-based). If none found, try stream.
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if len(tables) == 0:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for t in tables:
            # Use dataframe -> list of lists
            rows = t.df.values.tolist()
            # Camelot tables have page attribute (string)
            page = getattr(t, "page", None)
            results.append({"page": page, "type": "camelot", "data": rows})
    except Exception:
        # swallow exceptions; fallback will handle
        pass
    return results


def _compute_column_cuts(all_centers: List[float]) -> List[float]:
    """Compute split x positions (midpoints) using large gaps among centers."""
    if len(all_centers) < 2:
        return []

    arr = np.array(sorted(all_centers))
    diffs = np.diff(arr)
    if len(diffs) == 0:
        return []

    mean = np.mean(diffs)
    # dynamic threshold: gaps much larger than mean; tuned multiplier
    threshold = max(mean * 3.0, np.percentile(diffs, 75) * 1.5)
    split_indices = np.where(diffs > threshold)[0]
    cuts = []
    for idx in split_indices:
        cuts.append((arr[idx] + arr[idx + 1]) / 2.0)
    return cuts


def _group_words_into_rows(words: List[Dict]) -> List[List[Dict]]:
    """Group word boxes into rows by y coordinate proximity."""
    if not words:
        return []
    # sort by center_y
    words_sorted = sorted(words, key=lambda w: w["center_y"])
    heights = [w["height"] for w in words_sorted if w.get("height", 0) > 0]
    median_h = float(np.median(heights)) if heights else 10.0
    tol = max(10.0, median_h * 0.8)  # row tolerance

    rows = []
    current_row = {"y_sum": 0.0, "count": 0, "words": []}
    for w in words_sorted:
        cy = w["center_y"]
        if current_row["count"] == 0:
            current_row["y_sum"] = cy
            current_row["count"] = 1
            current_row["words"].append(w)
        else:
            avg_y = current_row["y_sum"] / current_row["count"]
            if abs(cy - avg_y) <= tol:
                current_row["y_sum"] += cy
                current_row["count"] += 1
                current_row["words"].append(w)
            else:
                rows.append(current_row["words"])
                current_row = {"y_sum": cy, "count": 1, "words": [w]}
    if current_row["count"] > 0:
        rows.append(current_row["words"])
    # sort words inside each row by left coordinate
    rows_sorted = [[dict(w) for w in sorted(row, key=lambda x: x["left"])] for row in rows]
    return rows_sorted


def _row_words_to_cells(rows: List[List[Dict]], cuts: List[float]) -> List[List[str]]:
    """Take grouped row words and cluster them into columns given cuts."""
    if not rows:
        return []

    # define bin edges using cuts: (-inf, cut1), (cut1, cut2), ..., (cutN, +inf)
    cuts_sorted = sorted(cuts)
    tables = []
    for row in rows:
        n_cols = len(cuts_sorted) + 1
        cells = [""] * n_cols
        for w in row:
            cx = w["center_x"]
            # find column index
            col_idx = 0
            while col_idx < len(cuts_sorted) and cx > cuts_sorted[col_idx]:
                col_idx += 1
            if cells[col_idx]:
                cells[col_idx] += " " + w["text"]
            else:
                cells[col_idx] = w["text"]
        # strip whitespace
        cells = [c.strip() for c in cells]
        tables.append(cells)
    return tables


def extract_tables_from_image(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Extract table-like structures from a PIL image using Tesseract word boxes and heuristics.
    Returns a list of tables: each is {"page": 1, "type": "ocr", "data": rows_as_list_of_lists}.
    """
    try:
        # ensure RGB
        img = pil_image.convert("RGB")
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception:
        return []

    n = len(data["text"])
    words = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if txt == "":
            continue
        try:
            conf = int(data["conf"][i])
        except Exception:
            conf = -1
        # accept most words (even low conf) because table structure matters
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        words.append({
            "text": txt,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "center_x": left + width / 2.0,
            "center_y": top + height / 2.0,
            "conf": conf,
            "line_num": int(data.get("line_num", [0]*n)[i])
        })

    if not words:
        return []

    # group words into rows
    rows = _group_words_into_rows(words)

    # compute global column cuts using centers of all words
    all_centers = [w["center_x"] for w in words]
    cuts = _compute_column_cuts(all_centers)

    table_rows = _row_words_to_cells(rows, cuts)
    return [{"page": 1, "type": "ocr", "data": table_rows}]


def extract_tables_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF bytes. Try Camelot first for digital PDFs.
    Fallback: convert pages to images and run image-based extraction.
    """
    results: List[Dict[str, Any]] = []

    # 1) Try camelot (requires a temporary file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmpf.write(pdf_bytes)
        tmp_path = tmpf.name

    if _HAS_CAMELOT:
        try:
            camelot_tables = _camelot_extract(tmp_path)
            if camelot_tables:
                # remove temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return camelot_tables
        except Exception:
            pass

    # 2) Fallback: convert PDF pages to images and apply OCR-based extraction
    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception:
        pil_pages = []

    for i, page_img in enumerate(pil_pages):
        page_tables = extract_tables_from_image(page_img)
        # tag actual page number
        for t in page_tables:
            t["page"] = i + 1
        results.extend(page_tables)

    # cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return results
