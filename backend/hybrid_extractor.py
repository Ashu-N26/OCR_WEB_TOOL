import os
import io
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import pdfplumber
import re
import tempfile
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes

# ============================================================
# ADVANCED POST-PROCESSOR FOR MERGING / CLEANING OCR TABLE DATA
# ============================================================

def postprocess_table(raw_rows, debug=False, expected_cols=None):
    if not raw_rows:
        return {"format": "empty", "data": []}

    # Normalize and clean rows
    cleaned = []
    for row in raw_rows:
        if isinstance(row, str):
            row = [row]
        row = [re.sub(r"\s+", " ", c.strip()) for c in row if c.strip()]
        if row:
            cleaned.append(row)

    # Detect single-column or multi-column layout
    col_counts = [len(r) for r in cleaned]
    if len(col_counts) == 0:
        return {"format": "empty", "data": []}
    median_cols = int(np.median(col_counts))

    # If too inconsistent, assume it's not a real table
    if median_cols <= 1:
        # Merge as simple paragraph text
        merged_lines = []
        buf = ""
        for r in cleaned:
            line = " ".join(r)
            if line.endswith("-"):
                buf += line[:-1]
            else:
                buf += line
                merged_lines.append(buf.strip())
                buf = ""
        if buf:
            merged_lines.append(buf.strip())
        return {"format": "lines", "data": merged_lines}

    # Merge similar rows and fix split cells
    merged = []
    prev = None
    for row in cleaned:
        if prev and len(row) == len(prev):
            # merge if they share structure and few words
            if sum(1 for a, b in zip(row, prev) if a == b) >= len(row) - 1:
                continue
        merged.append(row)
        prev = row

    # Deduplicate repeated headers (common OCR issue)
    seen_headers = set()
    filtered = []
    for row in merged:
        key = tuple(row)
        if key not in seen_headers:
            seen_headers.add(key)
            filtered.append(row)

    # Equalize column counts
    max_cols = expected_cols or median_cols
    normalized = []
    for r in filtered:
        if len(r) < max_cols:
            r += [""] * (max_cols - len(r))
        normalized.append(r[:max_cols])

    if debug:
        print("[PostProcessor] Cleaned rows:", len(normalized))
    return {"format": "table", "data": normalized}


# ============================================================
# MAIN HYBRID EXTRACTION PIPELINE
# ============================================================

def enhance_image(img):
    """Improve contrast and sharpness before OCR"""
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.convert("L").filter(ImageFilter.SHARPEN)
    return img


def extract_text_from_image(image):
    """Extract text with OCR using Tesseract"""
    config = "--psm 6"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()


def extract_tables_from_pdfplumber(pdf_bytes):
    """Try extracting tables directly using pdfplumber"""
    results = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    results.append(table)
    except Exception as e:
        print("[WARN] pdfplumber failed:", e)
    return results


def extract_tables_from_image_ocr(img):
    """Detect table-like patterns using OCR bounding boxes"""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    rows = []
    current_row = []
    last_y = None

    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if last_y is None:
            last_y = y
        # New line threshold
        if abs(y - last_y) > 15:
            rows.append(current_row)
            current_row = []
        current_row.append(word)
        last_y = y

    if current_row:
        rows.append(current_row)

    # Convert into list of lists (simple tabular form)
    rows = [r for r in rows if r]
    return rows


def extract_tables(file_content, filename, section_keyword=None, debug=False, dpi=300):
    """Hybrid extraction that handles PDF and Image uploads"""
    is_pdf = filename.lower().endswith(".pdf")

    # --------------------------
    # STEP 1: Convert to images
    # --------------------------
    images = []
    if is_pdf:
        try:
            images = convert_from_bytes(file_content, dpi=dpi)
        except Exception as e:
            print("[ERROR] PDF to image failed:", e)
    else:
        try:
            img = Image.open(io.BytesIO(file_content))
            images = [img]
        except Exception as e:
            print("[ERROR] Image open failed:", e)
            return {"error": str(e)}

    all_tables = []

    # --------------------------
    # STEP 2: OCR + Table detection
    # --------------------------
    for page_no, img in enumerate(images):
        enhanced = enhance_image(img)
        text = extract_text_from_image(enhanced)

        # Filter for specific section if keyword given
        if section_keyword:
            pattern = re.compile(re.escape(section_keyword), re.IGNORECASE)
            if not pattern.search(text):
                continue

        # Try direct PDF parsing (if applicable)
        if is_pdf:
            tables_pdf = extract_tables_from_pdfplumber(file_content)
            if tables_pdf:
                for t in tables_pdf:
                    all_tables.append(t)

        # Fallback to OCR-based extraction
        tables_ocr = extract_tables_from_image_ocr(enhanced)
        if tables_ocr:
            all_tables.extend(tables_ocr)

    # --------------------------
    # STEP 3: Post-processing
    # --------------------------
    if not all_tables:
        return {"format": "text", "data": ["No tables detected."]}

    processed = postprocess_table(all_tables, debug=debug)
    return processed


# ============================================================
# END OF MODULE
# ============================================================
