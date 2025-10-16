import os
import io
import cv2
import fitz
import pdfplumber
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from difflib import SequenceMatcher

# ==============================
# Utility Functions
# ==============================

def clean_text(text: str) -> str:
    """Clean OCR or extracted text."""
    if not text:
        return ""
    text = text.replace('\x0c', '').replace('\r', '')
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text


def merge_multiline_cells(text: str) -> str:
    """Merge lines that belong to the same cell or paragraph."""
    merged_lines = []
    buffer = ""
    for line in text.split("\n"):
        if line.endswith(("-", ":", ",")):
            buffer += line + " "
        elif buffer:
            merged_lines.append(buffer.strip() + " " + line)
            buffer = ""
        else:
            merged_lines.append(line)
    if buffer:
        merged_lines.append(buffer.strip())
    return "\n".join(merged_lines)


def remove_repeated_headers(lines):
    """Remove header lines repeated across pages."""
    if not lines:
        return lines
    result = [lines[0]]
    for line in lines[1:]:
        if not any(SequenceMatcher(None, line, prev).ratio() > 0.9 for prev in result[-3:]):
            result.append(line)
    return result


def postprocess_text(text: str) -> str:
    """Apply post-processing cleanup."""
    text = clean_text(text)
    text = merge_multiline_cells(text)
    lines = text.split("\n")
    lines = remove_repeated_headers(lines)
    return "\n".join(lines)


# ==============================
# PDF Extraction Core
# ==============================

def extract_text_from_pdf(pdf_path: str):
    """Extract text from PDF using pdfplumber."""
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    all_text += extracted + "\n"
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")
    return clean_text(all_text)


def extract_text_with_ocr(pdf_path: str):
    """OCR-based extraction using pytesseract."""
    all_text = ""
    try:
        pages = convert_from_path(pdf_path)
        for page in pages:
            img = np.array(page)
            text = pytesseract.image_to_string(img)
            all_text += text + "\n"
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
    return clean_text(all_text)


# ==============================
# Table Detection Logic
# ==============================

def detect_table_from_text(text: str):
    """Detect table-like structures from extracted text."""
    lines = text.split("\n")
    table_blocks, current_block = [], []

    for line in lines:
        if any(sep in line for sep in ["|", "\t"]) or (len(line.split()) > 4 and len(set(line)) > 5):
            current_block.append(line)
        elif current_block:
            table_blocks.append("\n".join(current_block))
            current_block = []
    if current_block:
        table_blocks.append("\n".join(current_block))

    tables = [block for block in table_blocks if len(block.split("\n")) > 1]
    return tables


def detect_table_from_image(pdf_path: str):
    """Detect table structures visually using OpenCV."""
    tables_detected = []
    try:
        pages = convert_from_path(pdf_path)
        for i, page in enumerate(pages):
            img = np.array(page)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
            )

            horizontal = thresh.copy()
            vertical = thresh.copy()

            cols = horizontal.shape[1]
            horizontal_size = cols // 30
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            rows = vertical.shape[0]
            vertical_size = rows // 30
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)

            mask = horizontal + vertical
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                tables_detected.append(f"Table structure visually detected on page {i+1}")
    except Exception as e:
        print(f"[ERROR] Table image detection failed: {e}")
    return tables_detected


# ==============================
# Section Filter & Table Formatter
# ==============================

def filter_section(text: str, keyword: str):
    """Extract specific section content if keyword provided."""
    if not keyword:
        return text
    lines = text.split("\n")
    extracted = []
    capture = False
    for line in lines:
        if keyword.lower() in line.lower():
            capture = True
        elif capture and (line.strip().startswith(tuple(str(i) for i in range(1, 10))) and "." in line[:5]):
            # Stop at next numbered section
            break
        if capture:
            extracted.append(line)
    return "\n".join(extracted) if extracted else text


def format_as_table_block(text_block: str):
    """Convert table-like text into HTML table."""
    rows = [r for r in text_block.split("\n") if r.strip()]
    html = "<table border='1' style='border-collapse:collapse;width:100%'>"
    for row in rows:
        html += "<tr>"
        for cell in row.split():
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</table>"
    return html


# ==============================
# Main Extraction Controller
# ==============================

def extract_tables(pdf_path: str, section_keyword: str = None):
    """Main orchestrator function."""
    if not os.path.exists(pdf_path):
        return {"error": "File not found"}

    # Step 1: Try text extraction
    text = extract_text_from_pdf(pdf_path)
    if not text:
        text = extract_text_with_ocr(pdf_path)

    # Step 2: Postprocess
    text = postprocess_text(text)

    # Step 3: Apply section filter (if any)
    if section_keyword:
        text = filter_section(text, section_keyword)

    # Step 4: Detect tables
    tables_from_text = detect_table_from_text(text)
    tables_from_image = detect_table_from_image(pdf_path)

    results = {
        "raw_text": text,
        "tables": [],
        "summary": {
            "total_tables_detected": len(tables_from_text) + len(tables_from_image),
            "detected_by_text": len(tables_from_text),
            "detected_by_image": len(tables_from_image),
        },
    }

    # Step 5: Convert detected tables into HTML
    for tbl in tables_from_text:
        formatted = format_as_table_block(tbl)
        results["tables"].append(formatted)

    if not results["tables"]:
        results["tables"].append("<p>No tables found, but text successfully extracted.</p>")

    return results
