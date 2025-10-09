# backend/table_extractor.py
"""
Advanced table extraction for OCR Web Tool.

Features:
 - PDF -> images conversion (pdf2image)
 - Adaptive preprocessing and morphological line detection (OpenCV)
 - Cell segmentation via contour detection and alignment
 - Per-cell OCR with confidence measurement (pytesseract)
 - Dynamic column clustering and grid reconstruction
 - Multiline / bilingual cell merging
 - Debug overlay image (base64) showing detected boxes & column lines
 - Logging for step-by-step diagnostics
"""

from io import BytesIO
import base64
import math
import logging

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
import pytesseract
from pytesseract import Output
import cv2

# Configure module logger (Render will capture stdout)
logger = logging.getLogger("table_extractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


# -------------------------
# Utilities
# -------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 300):
    """Convert PDF bytes to list of PIL RGB images."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        pages = [p.convert("RGB") for p in pages]
        logger.info("Converted PDF -> %d image(s) at %d DPI", len(pages), dpi)
        return pages
    except Exception as e:
        logger.exception("pdf_to_images failed: %s", e)
        return []


def pil_to_cv(img_pil: Image.Image):
    """Convert PIL Image (RGB) -> OpenCV BGR numpy array."""
    arr = np.array(img_pil)
    # convert RGB to BGR for OpenCV consistency
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_base64_png(img_cv):
    """Encode OpenCV BGR image to base64 PNG string."""
    _, buf = cv2.imencode(".png", img_cv)
    b64 = base64.b64encode(buf).decode("ascii")
    return b64


def clean_text(txt: str) -> str:
    if not txt:
        return ""
    s = str(txt).replace("\x0c", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    # Remove obvious OCR garbage tokens
    if len(s) <= 2 and s.lower() in {"me", "mr", "we", "mR".lower(), "lh"}:
        return ""
    return s


# -------------------------
# Preprocess image for line detection
# -------------------------
def preprocess_for_lines(img_cv, debug=False):
    """Return grayscale, thresh, horiz_mask, vert_mask"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # resize if very small (improve OCR)
    h, w = gray.shape
    scale = 1.0
    if max(h, w) < 1000:
        scale = 2.0
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        logger.info("Scaled image by %.2f for processing", scale)

    # denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # adaptive threshold - robust to lighting
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 9)

    # morphological kernels - sizes relative to width
    kernel_len_h = max(10, int(thr.shape[1] / 40))
    kernel_len_v = max(10, int(thr.shape[0] / 40))

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_v))

    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v, iterations=1)

    if debug:
        logger.info("Preprocess: image size=%s, kernel_h=%d kernel_v=%d", thr.shape, kernel_len_h, kernel_len_v)

    return {"gray": gray, "thresh": thr, "horiz": horiz, "vert": vert, "scale": scale}


# -------------------------
# Detect candidate cell bounding boxes
# -------------------------
def detect_cells(horiz_mask, vert_mask, min_area=500, debug_img=None):
    """
    Combine horizontal & vertical masks and find cell contours.
    Returns list of boxes (x,y,w,h).
    """
    # Combine masks: union gives full table grid; intersection gives grid intersections
    combined = cv2.add(horiz_mask, vert_mask)
    # optionally close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    # Optionally expand boxes slightly to include text near borders
    boxes2 = []
    for (x, y, w, h) in boxes:
        pad_x = int(max(2, w * 0.02))
        pad_y = int(max(2, h * 0.02))
        boxes2.append((max(0, x - pad_x), max(0, y - pad_y), w + 2 * pad_x, h + 2 * pad_y))

    # Sort top-to-bottom then left-to-right
    boxes_sorted = sorted(boxes2, key=lambda b: (b[1], b[0]))

    if debug_img is not None:
        vis = debug_img.copy()
        for (x, y, w, h) in boxes_sorted:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)
        return boxes_sorted, vis

    return boxes_sorted, None


# -------------------------
# Cluster columns by x position (1D clustering)
# -------------------------
def cluster_columns(boxes, x_tol=30):
    """
    Given boxes (x,y,w,h) across the page, cluster their x positions into columns.
    Returns list of column x-centers (sorted).
    """
    centers = sorted([x + w / 2.0 for x, y, w, h in boxes])
    if not centers:
        return []

    clusters = [[centers[0]]]
    for c in centers[1:]:
        if abs(c - clusters[-1][-1]) <= x_tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    col_centers = [int(sum(g) / len(g)) for g in clusters]
    return sorted(col_centers)


# -------------------------
# Build grid and OCR cells
# -------------------------
def build_grid_and_ocr(img_cv, boxes, col_centers, scale=1.0, ocr_conf_threshold=30):
    """
    Build a 2D grid: rows x cols and OCR each cell.
    Returns DataFrame-like list of rows and per-cell metadata (confidences).
    """
    # Group boxes into rows using y coordinate clustering
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    current_row = []
    last_y = None
    y_tol = max(10, int(np.median([h for _, _, _, h in boxes_sorted]) * 0.6)) if boxes_sorted else 20

    for b in boxes_sorted:
        x, y, w, h = b
        if last_y is None:
            current_row = [b]
            last_y = y
            continue
        if abs(y - last_y) <= y_tol:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda t: t[0]))
            current_row = [b]
            last_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t[0]))

    # Build matrix with columns = len(col_centers)
    num_cols = max(1, len(col_centers))
    grid = []
    meta = []  # store confs and bbox
    for r_idx, row in enumerate(rows):
        row_cells = [""] * num_cols
        row_meta = [None] * num_cols
        for (x, y, w, h) in row:
            # choose nearest col by distance to center
            cx = x + w / 2.0
            col_idx = int(np.argmin([abs(cx - cc) for cc in col_centers])) if col_centers else 0

            # expand region slightly to capture text
            ex, ey, ew, eh = max(0, x - 2), max(0, y - 2), min(img_cv.shape[1] - x + 2, w + 4), min(img_cv.shape[0] - y + 2, h + 4)
            crop = img_cv[ey:ey + eh, ex:ex + ew]
            # upscale small crops for better OCR
            ch, cw = crop.shape[:2]
            scale_factor = 1.0
            if max(ch, cw) < 60:
                scale_factor = 2.0
                crop = cv2.resize(crop, (int(cw * scale_factor), int(ch * scale_factor)), interpolation=cv2.INTER_LINEAR)

            # OCR with pytesseract: use image_to_data to get confidences
            try:
                data = pytesseract.image_to_data(crop, output_type=Output.DICT, config="--psm 6")
                # join text pieces
                text_parts = []
                confs = []
                n = len(data.get("text", []))
                for i in range(n):
                    t = (data["text"][i] or "").strip()
                    if t:
                        text_parts.append(t)
                        try:
                            confs.append(int(data.get("conf", [ -1 ])[i]))
                        except Exception:
                            confs.append(-1)
                text = " ".join(text_parts).strip()
                avg_conf = int(np.mean([c for c in confs if c >= 0]) ) if any(c >= 0 for c in confs) else -1
            except Exception as e:
                logger.exception("OCR cell failed: %s", e)
                text = ""
                avg_conf = -1

            text = clean_text(text)
            # if multiple detections into same column, append separated by space
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + text
            else:
                row_cells[col_idx] = text
            row_meta[col_idx] = {"bbox": (ex, ey, ew, eh), "conf": avg_conf}

        grid.append(row_cells)
        meta.append(row_meta)

    # Postprocess: merge adjacent empty columns, trim trailing/leading blanks
    df = pd.DataFrame(grid)
    # drop fully empty columns at edges
    # keep internal empty columns (they may be legitimate)
    # collapse repeated spaces
    df = df.fillna("").applymap(lambda x: " ".join(x.split()).strip())

    return df, meta, rows


# -------------------------
# Debug overlay: draw boxes and column lines
# -------------------------
def make_debug_overlay(img_cv, boxes, col_centers, rows_boxes=None):
    vis = img_cv.copy()
    # draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)
    # draw column centers as vertical lines
    for cc in col_centers:
        cv2.line(vis, (int(cc), 0), (int(cc), vis.shape[0]), (0, 200, 0), 1)
    # optionally draw row separators
    if rows_boxes:
        for r in rows_boxes:
            # compute y extents of row
            ys = [y for (_, y, _, _) in r]
            maxh = max([h for (_, _, _, h) in r])
            y0 = min(ys)
            cv2.line(vis, (0, y0), (vis.shape[1], y0), (200, 0, 0), 1)
    return vis


# -------------------------
# Public entry: extract_tables
# -------------------------
def extract_tables(file_bytes: bytes, filename: str, debug: bool = False):
    """
    Unified function to extract tables from either PDF or image bytes.

    Returns:
      {
        "html": <html table(s) string>,
        "json": [ list of row dicts ],
        "debug_image": <base64 PNG string>  # only if debug True
        "log": {.. statistics ..}
      }
    """
    logger.info("extract_tables called: %s (debug=%s)", filename, debug)
    pages = []
    if filename.lower().endswith(".pdf"):
        pages = pdf_to_images(file_bytes)
    else:
        try:
            pil = Image.open(BytesIO(file_bytes)).convert("RGB")
            pages = [pil]
        except Exception as e:
            logger.exception("Cannot open image: %s", e)
            return {"html": "", "json": [], "debug_image": None, "log": {"error": "cannot open image"}}

    all_html = []
    all_json = []
    debug_img_b64 = None
    stats = {"pages": len(pages), "page_stats": []}

    for i, pil_page in enumerate(pages, start=1):
        logger.info("Processing page %d/%d", i, len(pages))
        cv_img = pil_to_cv(pil_page)
        prep = preprocess_for_lines(cv_img, debug=debug)
        boxes, vis = detect_cells(prep["horiz"], prep["vert"], min_area=200, debug_img=cv_img.copy())
        logger.info("Detected %d candidate boxes on page %d", len(boxes), i)

        # If boxes small or none, optionally fallback to connected components on thresh
        if not boxes:
            logger.info("No boxes found - falling back to connected component detection on threshold")
            # find connected components on threshold image
            thr = prep["thresh"]
            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fallback_boxes = []
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 500:
                    continue
                fallback_boxes.append((x, y, w, h))
            boxes = sorted(fallback_boxes, key=lambda b: (b[1], b[0]))
            logger.info("Fallback detected %d boxes", len(boxes))

        # Column clustering
        col_centers = cluster_columns(boxes, x_tol=max(20, int(pil_page.size[0]/60)))
        logger.info("Column centers (n=%d): %s", len(col_centers), col_centers)

        # Build grid and OCR cells
        df, meta, rows_boxes = build_grid_and_ocr(cv_img, boxes, col_centers, scale=prep.get("scale", 1.0))
        logger.info("Built grid with %d rows x %d cols (before cleaning)", df.shape[0], df.shape[1])

        # Post-cleaning heuristics:
        # - Merge header lines (if first two rows are both header-like, combine)
        # - Remove rows with almost all empty
        # - Trim leading/trailing whitespace
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # Detect header merge scenario: if first two rows both have many strings and the second row looks like subheaders,
        # then combine them into a single header row.
        header_combined = False
        if df.shape[0] >= 2:
            first_nonnull = df.iloc[0].count()
            second_nonnull = df.iloc[1].count()
            if first_nonnull > 0 and second_nonnull > 0 and second_nonnull >= first_nonnull:
                # heuristic: combine into header row (join with " / ")
                hdr = []
                for c in range(df.shape[1]):
                    a = df.iat[0, c] if not pd.isna(df.iat[0, c]) else ""
                    b = df.iat[1, c] if not pd.isna(df.iat[1, c]) else ""
                    combined = " / ".join([x for x in [str(a).strip(), str(b).strip()] if x])
                    hdr.append(clean_text(combined))
                df = df.drop(index=[0, 1]).reset_index(drop=True)
                df.columns = range(df.shape[1])  # temp
                df_header = pd.DataFrame([hdr], columns=range(len(hdr)))
                df = pd.concat([df_header, df], ignore_index=True, axis=0)
                header_combined = True
                logger.info("Combined first two rows into header (heuristic)")

        # drop completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)
        # replace remaining NaN with empty string
        df = df.fillna("").applymap(lambda x: clean_text(x))

        # Convert DataFrame to HTML and JSON (list of rows)
        html = df.to_html(index=False, header=True, border=1, justify='left')
        json_rows = df.to_dict(orient="records")
        all_html.append(f"<h3>Page {i}</h3>\n{html}")
        all_json.append({"page": i, "rows": json_rows, "header_combined": header_combined})

        stats["page_stats"].append({
            "page": i,
            "boxes_detected": len(boxes),
            "cols_detected": len(col_centers),
            "rows_extracted": df.shape[0],
            "header_combined": header_combined
        })

        # Debug overlay (last page overwrites)
        if debug:
            overlay = make_debug_overlay(cv_img, boxes, col_centers, rows_boxes)
            debug_img_b64 = cv_to_base64_png(overlay)

    combined_html = "\n".join(all_html)
    # flatten json rows to single array for API convenience
    flattened_rows = []
    for p in all_json:
        for r in p["rows"]:
            flattened_rows.append({"page": p["page"], **r})

    result = {
        "html": combined_html,
        "json": flattened_rows,
        "debug_image": debug_img_b64,
        "log": stats
    }
    return result



