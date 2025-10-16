# backend/hybrid_extractor.py
import io
import re
import base64
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes
import cv2

logger = logging.getLogger("hybrid_extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------------
# Utilities
# -------------------------
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x0c", "").replace("\r", "")
    lines = [ln.rstrip() for ln in text.split("\n")]
    # drop trailing/leading empty lines but preserve internal blank lines if useful
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def ocr_image_to_text(img: Image.Image) -> str:
    # accept PIL image
    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        logger.exception("pytesseract failed: %s", e)
        text = ""
    return clean_text(text)


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF bytes to list of PIL Images using pdf2image."""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        return imgs
    except Exception as e:
        logger.exception("convert_from_bytes failed: %s", e)
        return []


# -------------------------
# Text extraction
# -------------------------
def extract_text_pages(pdf_bytes: bytes) -> List[str]:
    """Try pdfplumber (text layer) per page. Return list of page texts."""
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                texts.append(clean_text(txt))
        return texts
    except Exception as e:
        logger.warning("pdfplumber text extraction failed: %s", e)
        return []


def ocr_pages_from_images(images: List[Image.Image]) -> List[str]:
    texts = []
    for img in images:
        texts.append(ocr_image_to_text(img))
    return texts


# -------------------------
# Text-based table detection & reconstruction
# -------------------------
def is_table_like_line(line: str) -> bool:
    """Heuristic to decide if a line looks tabular: multiple big spaces or pipes or tabs."""
    if not line or len(line.strip()) == 0:
        return False
    # contains pipe or tab
    if "|" in line or "\t" in line:
        return True
    # large gaps / multiple consecutive spaces indicates columns
    # count sequences of 2+ spaces
    if len(re.findall(r" {2,}", line)) >= 1:
        return True
    # if many short tokens but consistent spacing maybe tabular
    tokens = line.split()
    return len(tokens) >= 3 and len(line) / max(1, len(tokens)) > 6


def group_table_blocks(page_text: str) -> List[str]:
    """
    From a page's text, return blocks (strings) that look like tables.
    Groups consecutive table-like lines.
    """
    if not page_text:
        return []
    lines = page_text.splitlines()
    blocks = []
    current = []
    for ln in lines:
        if is_table_like_line(ln):
            current.append(ln)
        else:
            if current:
                blocks.append("\n".join(current))
                current = []
    if current:
        blocks.append("\n".join(current))
    # keep blocks with at least 2 rows
    return [b for b in blocks if len(b.splitlines()) >= 2]


def compute_column_boundaries(lines: List[str], min_gap: int = 3) -> List[int]:
    """
    Analyze character occupancy matrix to identify column boundaries.
    Return list of split indices (end positions for columns).
    Approach:
      - Pad lines to same length
      - Build occupancy array across rows: 1 if char is non-space
      - Sum occupancy vertically: high occupancy -> content column; zero -> gap
      - Find runs of gaps longer than min_gap => split between columns
    """
    if not lines:
        return []

    max_len = max(len(ln) for ln in lines)
    matrix = np.zeros((len(lines), max_len), dtype=np.uint8)
    for i, ln in enumerate(lines):
        for j, ch in enumerate(ln.ljust(max_len)):
            matrix[i, j] = 1 if (ch != " ") else 0
    col_occupancy = matrix.sum(axis=0)  # how many rows have a char at each column pos

    # consider a column boundary where occupancy is low across rows.
    boundaries = []
    j = 0
    while j < max_len:
        if col_occupancy[j] == 0:
            # run of zeros
            start = j
            while j < max_len and col_occupancy[j] == 0:
                j += 1
            gap_len = j - start
            if gap_len >= min_gap:
                # boundary resides at middle of gap: use start index as split
                boundaries.append(start)
        else:
            j += 1
    # remove duplicates & ensure not including 0 or max_len
    bounds = sorted(set([b for b in boundaries if 0 < b < max_len]))
    return bounds


def split_line_by_bounds(line: str, bounds: List[int]) -> List[str]:
    """Split a single line using column boundaries indices."""
    if not bounds:
        return [line.strip()]
    max_len = max(len(line), bounds[-1] + 1 if bounds else len(line))
    s = line.ljust(max_len)
    parts = []
    prev = 0
    for b in bounds:
        parts.append(s[prev:b].strip())
        prev = b
    parts.append(s[prev:].strip())
    return parts


def reconstruct_table_from_block(block_text: str) -> Optional[pd.DataFrame]:
    """
    Given a text block which is tabular-like (several lines),
    reconstruct a DataFrame using automatic column boundary detection.
    """
    lines = [ln for ln in block_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    # Some blocks may use '|' or '\t' to separate; prefer these simple splits if present
    if all("|" in ln for ln in lines):
        rows = [ [c.strip() for c in ln.split("|") if True] for ln in lines ]
        # normalize column count
        maxcols = max(len(r) for r in rows)
        rows = [r + [""] * (maxcols - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 1 else [f"col{i}" for i in range(maxcols)])
        return df

    if any("\t" in ln for ln in lines):
        rows = [ [c.strip() for c in ln.split("\t")] for ln in lines ]
        maxcols = max(len(r) for r in rows)
        rows = [r + [""] * (maxcols - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 1 else [f"col{i}" for i in range(maxcols)])
        return df

    # Fallback: compute column boundaries from occupancy
    bounds = compute_column_boundaries(lines, min_gap=2)
    rows = [split_line_by_bounds(ln, bounds) for ln in lines]

    # Some rows may have different column counts — normalize
    maxcols = max(len(r) for r in rows)
    norm_rows = [r + [""] * (maxcols - len(r)) for r in rows]

    # Decide headers: if first row words short and second row numeric lengths -> first row header likely
    header = None
    if len(norm_rows) >= 2:
        # header heuristics: if first row has alpha-heavy cells and subsequent row numeric or mix, choose header
        first_alpha = sum([bool(re.search(r"[A-Za-z]", c)) for c in norm_rows[0]])
        second_alpha = sum([bool(re.search(r"[A-Za-z]", c)) for c in norm_rows[1]])
        if first_alpha >= second_alpha:
            header = norm_rows[0]
            data_rows = norm_rows[1:]
        else:
            header = [f"col{i+1}" for i in range(maxcols)]
            data_rows = norm_rows
    else:
        header = [f"col{i+1}" for i in range(maxcols)]
        data_rows = norm_rows

    df = pd.DataFrame(data_rows, columns=header)
    # drop fully-empty columns
    df = df.loc[:, ~(df == "").all(axis=0)]
    if df.shape[1] == 0:
        return None
    return df


# -------------------------
# Image-based table detection fallback
# -------------------------
def detect_visual_table_regions(image: Image.Image, min_area: int = 2000) -> List[Dict[str,int]]:
    """
    Given PIL image, return list of bounding boxes where a table-like box exists.
    Uses morphological operators to find grid-like areas.
    Returns list of dicts {x, y, w, h}
    """
    boxes = []
    try:
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # invert & threshold
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # morphological to detect lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        mask = cv2.add(horizontal, vertical)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h >= min_area:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    except Exception as e:
        logger.exception("visual detection failed: %s", e)
    return boxes


def ocr_crop_and_reconstruct(image: Image.Image, box: Dict[str,int]) -> Optional[pd.DataFrame]:
    """Crop region, OCR it, then run reconstruct_table_from_block on OCR text."""
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cropped = image.crop((x, y, x + w, y + h))
    text = ocr_image_to_text(cropped)
    if not text or len(text.strip()) == 0:
        return None
    return reconstruct_table_from_block(text)


# -------------------------
# Export utilities
# -------------------------
def dataframe_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=1)

def dataframe_to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)

def dataframe_to_xlsx_base64(df: pd.DataFrame) -> str:
    out = io.BytesIO()
    # Use openpyxl if available
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        out.seek(0)
        return base64.b64encode(out.read()).decode("utf-8")
    except Exception as e:
        logger.exception("Failed to write xlsx: %s", e)
        return ""


# -------------------------
# Main public orchestrator
# -------------------------
def extract_tables(pdf_bytes: bytes,
                   filename: Optional[str] = None,
                   section_keyword: Optional[str] = None,
                   dpi: int = 300,
                   debug: bool = False) -> Dict[str, Any]:
    """
    High-level function:
      - Always extract per-page text (pdfplumber), fallback to OCR images when missing.
      - For each page try:
         a) detect table blocks purely from text and reconstruct columns
         b) if not found: visually detect table regions -> OCR those regions -> reconstruct
      - Return structured JSON with raw_text, page-level tables, and downloadable assets (csv/xlsx base64)
    """
    result = {
        "filename": filename,
        "pages": [],
        "summary": {"pages_scanned": 0, "tables_found": 0, "notes": []}
    }

    # 1) text-layer extraction
    page_texts = extract_text_pages(pdf_bytes)
    images = []
    # If text extraction returned nothing, do OCR on images (and also keep images for visual fallback)
    if not page_texts or all(not t.strip() for t in page_texts):
        result["summary"]["notes"].append("No PDF text layer found; running OCR on images for all pages.")
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        page_texts = ocr_pages_from_images(images)
    else:
        # still prepare images for visual fallback if needed
        images = pdf_to_images(pdf_bytes, dpi=dpi)

    result["summary"]["pages_scanned"] = len(page_texts)

    # 2) per-page analysis
    for page_idx, page_text in enumerate(page_texts):
        page_record: Dict[str, Any] = {"page": page_idx + 1, "raw_text": page_text, "tables": []}
        # if user requested a specific section keyword, filter page_text first
        if section_keyword:
            if section_keyword.lower() not in page_text.lower():
                # skip page if section keyword not present
                # but we will still include raw_text for data-first workflow
                result["pages"].append(page_record)
                continue
            else:
                page_record["filtered_by_section"] = True
                result["summary"]["notes"].append(f"Page {page_idx+1} matched section keyword '{section_keyword}'.")

        # 2a) text-based detection
        text_blocks = group_table_blocks(page_text)
        reconstructed_any = False
        for block in text_blocks:
            df = reconstruct_table_from_block(block)
            if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                reconstructed_any = True
                html = dataframe_to_html(df)
                csv_text = dataframe_to_csv_text(df)
                xlsx_b64 = dataframe_to_xlsx_base64(df)
                page_record["tables"].append({
                    "method": "text",
                    "html": html,
                    "csv": csv_text,
                    "xlsx_base64": xlsx_b64,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1])
                })

        # 2b) visual detection fallback if no text-based tables
        if not reconstructed_any and page_idx < len(images):
            boxes = detect_visual_table_regions(images[page_idx], min_area=1500)
            found_v = False
            for box in boxes:
                df = ocr_crop_and_reconstruct(images[page_idx], box)
                if df is not None and df.shape[0] > 0:
                    found_v = True
                    html = dataframe_to_html(df)
                    csv_text = dataframe_to_csv_text(df)
                    xlsx_b64 = dataframe_to_xlsx_base64(df)
                    page_record["tables"].append({
                        "method": "visual_ocr",
                        "bbox": box,
                        "html": html,
                        "csv": csv_text,
                        "xlsx_base64": xlsx_b64,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1])
                    })
            if found_v:
                reconstructed_any = True

        if reconstructed_any:
            result["summary"]["tables_found"] += len(page_record["tables"])
        result["pages"].append(page_record)

    # 3) if no tables at all, still provide cleaned text
    if result["summary"]["tables_found"] == 0:
        result["summary"]["notes"].append("No tables found; returning cleaned extracted text only.")
    else:
        result["summary"]["notes"].append(f"{result['summary']['tables_found']} table(s) detected and reconstructed.")

    if debug:
        result["debug_info"] = {"page_text_lengths": [len(p["raw_text"]) for p in result["pages"]],
                                "pages_count": result["summary"]["pages_scanned"]}

    return result
