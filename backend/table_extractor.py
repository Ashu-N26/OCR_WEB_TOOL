# backend/table_extractor.py
import io
import tempfile
import os
from typing import List, Dict, Any

from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np

# pdfplumber for table extraction from digital PDFs
import pdfplumber
from pdf2image import convert_from_bytes

# --- Helpers: group words into rows and columns (heuristic) --- #
def _group_words_into_rows(words: List[Dict]) -> List[List[Dict]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: w["center_y"])
    heights = [w["height"] for w in words_sorted if w.get("height", 0) > 0]
    median_h = float(np.median(heights)) if heights else 10.0
    tol = max(10.0, median_h * 0.8)
    rows = []
    current = {"y_sum": 0.0, "count": 0, "words": []}
    for w in words_sorted:
        cy = w["center_y"]
        if current["count"] == 0:
            current = {"y_sum": cy, "count": 1, "words": [w]}
        else:
            avg_y = current["y_sum"] / current["count"]
            if abs(cy - avg_y) <= tol:
                current["y_sum"] += cy
                current["count"] += 1
                current["words"].append(w)
            else:
                rows.append(current["words"])
                current = {"y_sum": cy, "count": 1, "words": [w]}
    if current["count"] > 0:
        rows.append(current["words"])
    # sort words inside each row by left coordinate
    return [[dict(w) for w in sorted(r, key=lambda x: x["left"])] for r in rows]

def _compute_column_cuts(all_centers: List[float]) -> List[float]:
    if len(all_centers) < 2:
        return []
    arr = np.array(sorted(all_centers))
    diffs = np.diff(arr)
    if len(diffs) == 0:
        return []
    mean = np.mean(diffs)
    threshold = max(mean * 3.0, np.percentile(diffs, 75) * 1.5)
    split_indices = np.where(diffs > threshold)[0]
    cuts = []
    for idx in split_indices:
        cuts.append((arr[idx] + arr[idx + 1]) / 2.0)
    return cuts

def _row_words_to_cells(rows: List[List[Dict]], cuts: List[float]) -> List[List[str]]:
    if not rows:
        return []
    cuts_sorted = sorted(cuts)
    tables = []
    for row in rows:
        n_cols = len(cuts_sorted) + 1
        cells = [""] * n_cols
        for w in row:
            cx = w["center_x"]
            col_idx = 0
            while col_idx < len(cuts_sorted) and cx > cuts_sorted[col_idx]:
                col_idx += 1
            if cells[col_idx]:
                cells[col_idx] += " " + w["text"]
            else:
                cells[col_idx] = w["text"]
        tables.append([c.strip() for c in cells])
    return tables

# --- Main functions --- #
def extract_tables_from_image(pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    OCR-based table detection from an image (PIL Image).
    Returns list of table dicts: {"page": 1, "type": "ocr", "data": rows}
    """
    try:
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
            "conf": conf
        })

    if not words:
        return []

    rows = _group_words_into_rows(words)
    all_centers = [w["center_x"] for w in words]
    cuts = _compute_column_cuts(all_centers)
    table_rows = _row_words_to_cells(rows, cuts)
    return [{"page": 1, "type": "ocr", "data": table_rows}]

def extract_tables_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Try pdfplumber (digital PDF) to extract tables. If none found, convert pages to images and use OCR method.
    Returns list of table dicts.
    """
    results = []

    # 1) Try pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        if tbl and any(any(cell for cell in row) for row in tbl):
                            results.append({"page": i, "type": "pdfplumber", "data": tbl})
                except Exception:
                    # skip page table errors
                    continue
    except Exception:
        # pdfplumber may fail on some PDFs; fallback below
        pass

    if results:
        return results

    # 2) Fallback: convert pages to images and apply OCR-based extraction
    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception:
        pil_pages = []

    for i, page_img in enumerate(pil_pages, start=1):
        page_tables = extract_tables_from_image(page_img)
        for t in page_tables:
            t["page"] = i
        results.extend(page_tables)

    return results

