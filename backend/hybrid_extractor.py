# backend/hybrid_extractor.py
"""
Hybrid extractor for OCR Web Tool.

Exports:
    extract_tables(file_bytes: bytes, filename: str, section_keyword: str = "2.14",
                   extra_keywords: Optional[List[str]] = None, debug: bool = False, dpi: int = 300) -> dict

Returns a dict:
{
  "pages": [
    {
      "page": <int>,
      "method": "pdf_text"|"ocr_page"|"pdfplumber_table"|"image_table"|"ocr_fallback",
      "data": [ [cell1, cell2, ...], ... ],   # list of rows (strings)
      "html": "<table>...</table>",
      "debug_image": "<base64 png>" | None,
      "meta": {...}
    },
    ...
  ],
  "summary": {...}
}
"""
from io import BytesIO
import base64
import logging
import re
from typing import List, Tuple, Optional, Dict, Any

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

from difflib import SequenceMatcher

logger = logging.getLogger("hybrid_extractor")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Utilities
# -----------------------
def _pil_to_cv(pil_img: "Image.Image"):
    """Convert PIL Image to OpenCV BGR (numpy)."""
    if Image is None:
        raise RuntimeError("Pillow is required")
    arr = np.array(pil_img)
    # Pillow gives RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_pil(cv_img: "np.ndarray"):
    arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def _img_to_b64png(img_cv: "np.ndarray") -> str:
    """Return base64 PNG of OpenCV image."""
    _, buf = cv2.imencode(".png", img_cv)
    return base64.b64encode(buf).decode("ascii")


def _clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x0c", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.65) -> bool:
    """Return True if needle is close to any substring of haystack using SequenceMatcher.
    This is used for tolerant section detection (handles '2 . 14' vs '2.14', OCR quirks).
    """
    if not haystack or not needle:
        return False
    hay = haystack.lower()
    needle = needle.lower()
    # quick exact containment
    if needle in hay:
        return True
    # sliding window over tokens for fuzzy match
    tokens = hay.split()
    n_tokens = len(tokens)
    needle_tokens = needle.split()
    m = len(needle_tokens)
    if m == 0:
        return False
    # try windows sized from m to m+3 to match surrounding words
    for window in range(m, min(n_tokens, m + 4) + 1):
        for i in range(0, n_tokens - window + 1):
            sub = " ".join(tokens[i:i + window])
            ratio = SequenceMatcher(None, sub, needle).ratio()
            if ratio >= threshold:
                return True
    # last resort compare whole strings similarity
    if SequenceMatcher(None, hay, needle).ratio() >= threshold:
        return True
    return False


# -----------------------
# Keyword search helpers
# -----------------------
def _normalize_keyword_list(section_keyword: str, extra_keywords: Optional[List[str]] = None) -> List[str]:
    keywords = [section_keyword] if section_keyword else []
    if extra_keywords:
        keywords += extra_keywords
    # Add common variants to help detection
    variants = [
        section_keyword,
        section_keyword.replace(".", " . "),
        section_keyword.replace(".", ""),
        f"{section_keyword} approach",
        "approach",
        "aproximação",
        "approach and runway lighting",
        "runway lighting",
        "approach and runway",
        "ad " + section_keyword,
    ]
    for v in variants:
        if v and v not in keywords:
            keywords.append(v)
    # normalize
    return [k.strip() for k in keywords if k]


# -----------------------
# PDF text-layer based keyword search (PyMuPDF)
# -----------------------
def _find_keyword_in_pdf(doc_bytes: bytes, keywords: List[str]) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Return list of (page_num (1-based), bbox in PDF points (x0,y0,x1,y1)) for matches."""
    results = []
    if fitz is None:
        logger.debug("PyMuPDF not installed; skipping PDF text-layer search")
        return results
    try:
        doc = fitz.open(stream=doc_bytes, filetype="pdf")
    except Exception as e:
        logger.exception("fitz.open failed: %s", e)
        return results

    for i, page in enumerate(doc, start=1):
        try:
            words = page.get_text("words")  # list of tuples (x0,y0,x1,y1,word,block_no,line_no,word_no)
            if not words:
                # Try page.get_text("text") fallback
                text = page.get_text("text") or ""
                text_lower = text.lower()
                for kw in keywords:
                    if _fuzzy_contains(text_lower, kw.lower(), threshold=0.7):
                        # don't have coordinates: return full-page bbox
                        results.append((i, (0.0, 0.0, page.rect.width, page.rect.height)))
                        break
                continue

            # Build string and mapping of index -> bbox
            # words is list of tuples
            word_strs = [w[4] for w in words]
            # sliding window up to 6 tokens to detect multi-token keywords
            n = len(words)
            for kw in keywords:
                kw_l = kw.lower()
                # exact search on joined page text
                joined = " ".join(word_strs).lower()
                if kw_l in joined:
                    # find approximate word sequence
                    for start in range(0, n):
                        for end in range(start, min(n, start + 8)):
                            seq = " ".join(word_strs[start:end + 1]).lower()
                            if kw_l in seq or _fuzzy_contains(seq, kw_l, threshold=0.75):
                                # compute bbox spanning words[start:end]
                                x0 = min(words[k][0] for k in range(start, end + 1))
                                y0 = min(words[k][1] for k in range(start, end + 1))
                                x1 = max(words[k][2] for k in range(start, end + 1))
                                y1 = max(words[k][3] for k in range(start, end + 1))
                                results.append((i, (x0, y0, x1, y1)))
                                raise StopIteration
                    # if not found continue
            # Also try fuzzy window search if nothing found
            if not any(r[0] == i for r in results):
                for start in range(0, n):
                    for end in range(start, min(n, start + 8)):
                        seq = " ".join(word_strs[start:end + 1]).lower()
                        for kw in keywords:
                            if _fuzzy_contains(seq, kw.lower(), threshold=0.82):
                                x0 = min(words[k][0] for k in range(start, end + 1))
                                y0 = min(words[k][1] for k in range(start, end + 1))
                                x1 = max(words[k][2] for k in range(start, end + 1))
                                y1 = max(words[k][3] for k in range(start, end + 1))
                                results.append((i, (x0, y0, x1, y1)))
                                raise StopIteration
        except StopIteration:
            continue
        except Exception:
            logger.exception("Exception scanning page %s for keywords", i)
            continue
    try:
        doc.close()
    except Exception:
        pass
    return results


# -----------------------
# OCR-based keyword search on an image
# -----------------------
def _find_keyword_in_image(pil_img: "Image.Image", keywords: List[str]) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """Return list of ((x, y, w, h), matched_keyword) in image pixels."""
    out = []
    if pytesseract is None or Image is None:
        logger.debug("pytesseract or Pillow not available; cannot OCR image for keyword")
        return out
    try:
        cv = _pil_to_cv(pil_img)
        odata = pytesseract.image_to_data(cv, output_type=Output.DICT, config="--psm 6")
    except Exception:
        logger.exception("pytesseract.image_to_data failed")
        return out

    texts = odata.get("text", [])
    for i, txt in enumerate(texts):
        if not txt or not txt.strip():
            continue
        txt_norm = txt.strip()
        for kw in keywords:
            if _fuzzy_contains(txt_norm, kw, threshold=0.6) or kw.lower() in txt_norm.lower():
                # use this single-word bbox
                try:
                    x = int(odata["left"][i])
                    y = int(odata["top"][i])
                    w = int(odata["width"][i])
                    h = int(odata["height"][i])
                    out.append(((x, y, w, h), kw))
                except Exception:
                    continue
    # also try multi-word sequences by concatenating neighboring tokens (sliding window)
    if not out:
        toks = []
        n = len(texts)
        for i in range(n):
            toks.append({
                "text": texts[i],
                "left": int(odata["left"][i]),
                "top": int(odata["top"][i]),
                "width": int(odata["width"][i]),
                "height": int(odata["height"][i])
            })
        for kw in keywords:
            kw_l = kw.lower()
            for start in range(0, max(0, n - 1)):
                for end in range(start, min(n - 6, n - 1) + 1):
                    seq = " ".join([toks[k]["text"] for k in range(start, end + 1)]).strip().lower()
                    if _fuzzy_contains(seq, kw_l, threshold=0.7) or kw_l in seq:
                        x0 = min(toks[k]["left"] for k in range(start, end + 1))
                        y0 = min(toks[k]["top"] for k in range(start, end + 1))
                        x1 = max(toks[k]["left"] + toks[k]["width"] for k in range(start, end + 1))
                        y1 = max(toks[k]["top"] + toks[k]["height"] for k in range(start, end + 1))
                        out.append(((x0, y0, x1 - x0, y1 - y0), kw))
                        break
                if out:
                    break
            if out:
                break
    return out


# -----------------------
# Image preprocessing and optional deskew
# -----------------------
def _preprocess_for_table(cv_img: "np.ndarray", max_dim: int = 2000) -> "np.ndarray":
    """Resize, convert to gray, denoise, adaptive threshold. Returns binary image and grayscale copy."""
    if cv_img is None:
        raise RuntimeError("cv_img is None")
    h, w = cv_img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = cv_img.shape[:2]

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # optional denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # adaptive threshold - invert so foreground white on black (helpful for morphological ops)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 9)
    return bin_img, gray, cv_img


# -----------------------
# Table detection from image region
# -----------------------
def _extract_table_from_image_region(pil_img: "Image.Image", debug: bool = False) -> Dict[str, Any]:
    """Detect table grid + OCR each cell. Return dict with data + html + debug_image."""
    result = {"method": "ocr_fallback", "data": [], "html": "", "debug_image": None, "rows": 0, "cols": 0}
    if cv2 is None or pytesseract is None or np is None or Image is None:
        result["summary"] = {"error": "missing dependencies: opencv / pytesseract / numpy / Pillow required"}
        return result

    img_cv = _pil_to_cv(pil_img)
    bin_img, gray, orig_img = _preprocess_for_table(img_cv)

    h, w = bin_img.shape[:2]
    # morphological operations to detect lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 20), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))

    horiz_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    mask = cv2.add(horiz_lines, vert_lines)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww * hh < 800:  # ignore tiny
            continue
        boxes.append((x, y, ww, hh))

    # If we found boxes that look like table cells, build grid
    if boxes and len(boxes) >= 4:
        # sort boxes into rows by y coordinate
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

        # Compute columns by center x from the largest row
        col_centers = []
        if rows:
            largest_row = max(rows, key=lambda r: len(r))
            centers = [int(x + wbox / 2) for (x, y, wbox, hbox) in largest_row]
            col_centers = sorted(centers)

        # Build grid by mapping each box to nearest column center
        grid = []
        for r in rows:
            # create empty row cells with same number of columns
            row_cells = [""] * max(1, len(col_centers))
            for (x, y, ww, hh) in r:
                cx = int(x + ww / 2)
                if not col_centers:
                    col_idx = 0
                else:
                    # nearest center index
                    col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
                # crop cell with margin
                ex, ey = max(0, x - 2), max(0, y - 2)
                ew, eh = min(ww + 4, orig_img.shape[1] - ex), min(hh + 4, orig_img.shape[0] - ey)
                crop = orig_img[ey:ey + eh, ex:ex + ew]
                # OCR the crop
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

        df = pd.DataFrame(grid).replace(r'^\s*$', "", regex=True)
        result["method"] = "image_table_lines"
        result["data"] = df.to_numpy().astype(str).tolist()
        result["html"] = df.to_html(index=False, header=False, border=1)
        result["rows"], result["cols"] = df.shape
        if debug:
            overlay = orig_img.copy()
            for (x, y, ww, hh) in boxes:
                cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (0, 128, 255), 2)
            if col_centers:
                for c in col_centers:
                    cv2.line(overlay, (int(c), 0), (int(c), overlay.shape[0]), (0, 200, 0), 1)
            result["debug_image"] = _img_to_b64png(overlay)
        return result

    # If no clear lines/cells found, try OCR word-clustering approach
    try:
        cv_img = orig_img
        odata = pytesseract.image_to_data(cv_img, output_type=Output.DICT, config="--psm 6")
        words = []
        n = len(odata.get("text", []))
        for i in range(n):
            t = (odata["text"][i] or "").strip()
            if not t:
                continue
            conf = int(odata.get("conf", [-1])[i]) if odata.get("conf") else -1
            if conf < 10:
                # still keep low-confidence tokens but mark
                pass
            left = int(odata.get("left", [0])[i])
            top = int(odata.get("top", [0])[i])
            width = int(odata.get("width", [0])[i])
            height = int(odata.get("height", [0])[i])
            words.append({"text": t, "left": left, "top": top, "width": width, "height": height, "conf": conf})
    except Exception:
        logger.exception("pytesseract.image_to_data failed - performing full-image OCR fallback")
        try:
            text = pytesseract.image_to_string(orig_img, config="--psm 6")
            rows = [r.strip() for r in text.splitlines() if r.strip()]
            result["method"] = "ocr_fallback"
            result["data"] = [[r] for r in rows]
            result["html"] = pd.DataFrame(result["data"]).to_html(index=False, header=False, border=1)
            result["rows"], result["cols"] = len(result["data"]), 1
            return result
        except Exception as e:
            result["summary"] = {"error": f"full OCR fallback failed: {e}"}
            return result

    # cluster words into lines (rows) by y coordinate
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
                # update last_y as median
                last_y = int(sum([x["top"] for x in current_row]) / len(current_row))
            else:
                rows_clusters.append(sorted(current_row, key=lambda x: x["left"]))
                current_row = [w]
                last_y = w["top"]
    if current_row:
        rows_clusters.append(sorted(current_row, key=lambda x: x["left"]))

    # determine column boundaries by collecting x-centers across all rows and clustering
    centers = []
    for r in rows_clusters:
        for w in r:
            centers.append(w["left"] + w["width"] / 2.0)
    centers = sorted(centers)
    # cluster centers with a tolerance
    col_centers = []
    tol = max(20, int(orig_img.shape[1] / 40))
    if centers:
        group = [centers[0]]
        for c in centers[1:]:
            if abs(c - group[-1]) <= tol:
                group.append(c)
            else:
                col_centers.append(int(sum(group) / len(group)))
                group = [c]
        col_centers.append(int(sum(group) / len(group)))
    if not col_centers:
        # fallback single column
        col_centers = [int(orig_img.shape[1] / 2)]

    # build rows of text aligned to column centers
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

    # normalize row lengths (pad)
    maxcols = max(len(r) for r in grid) if grid else len(col_centers)
    for r in grid:
        while len(r) < maxcols:
            r.append("")

    df = pd.DataFrame(grid).replace(r'^\s*$', "", regex=True)
    result["method"] = "image_table_ocr_cluster"
    result["data"] = df.to_numpy().astype(str).tolist()
    result["html"] = df.to_html(index=False, header=False, border=1)
    result["rows"], result["cols"] = df.shape
    if debug:
        overlay = orig_img.copy()
        for w in words_sorted:
            cv2.rectangle(overlay, (w["left"], w["top"]), (w["left"] + w["width"], w["top"] + w["height"]), (255, 0, 0), 1)
        for c in col_centers:
            cv2.line(overlay, (int(c), 0), (int(c), overlay.shape[0]), (0, 255, 0), 1)
        result["debug_image"] = _img_to_b64png(overlay)
    return result


# -----------------------
# Main public function
# -----------------------
def extract_tables(file_bytes: bytes,
                   filename: str,
                   section_keyword: str = "2.14",
                   extra_keywords: Optional[List[str]] = None,
                   debug: bool = False,
                   dpi: int = 300) -> Dict[str, Any]:
    """
    Unified entry point.
    """
    keywords = _normalize_keyword_list(section_keyword, extra_keywords)
    logger.info("Searching for keywords: %s", keywords)

    results_pages: List[Dict[str, Any]] = []
    summary = {"pages_found": 0, "matches": 0, "methods": {}}

    # Detect file extension
    fname = (filename or "file").lower()
    ext = fname.split(".")[-1] if "." in fname else ""

    # --- PDF path: try PyMuPDF text search first (vector PDFs) ---
    if ext == "pdf" and fitz is not None:
        try:
            matches = _find_keyword_in_pdf(file_bytes, keywords)
        except Exception:
            matches = []
            logger.exception("Error while searching PDF text layer")
        if matches:
            logger.info("Keyword matches in PDF text layer: %s", matches)
            for page_num, bbox in matches:
                # convert page to image
                if convert_from_bytes is None or Image is None:
                    logger.warning("pdf2image or Pillow not available; cannot rasterize PDF page")
                    continue
                pil_page = None
                try:
                    imgs = convert_from_bytes(file_bytes, dpi=dpi, first_page=page_num, last_page=page_num)
                    if imgs:
                        pil_page = imgs[0].convert("RGB")
                except Exception:
                    logger.exception("pdf2image conversion failed for page %s", page_num)
                    pil_page = None

                if pil_page is None:
                    continue

                # bbox in PDF points; convert to pixels
                x0, y0, x1, y1 = bbox
                scale = dpi / 72.0
                px0 = int(max(0, x0 * scale))
                py0 = int(max(0, y0 * scale))
                px1 = int(min(pil_page.width, x1 * scale))
                py1 = int(min(pil_page.height, y1 * scale))
                # expand crop to include the table area below/around header
                pad_w = int(pil_page.width * 0.05)
                pad_h = int(pil_page.height * 0.18)
                crop_x0 = max(0, px0 - pad_w)
                crop_y0 = max(0, py0 - pad_h)
                crop_x1 = min(pil_page.width, px1 + pad_w)
                crop_y1 = min(pil_page.height, py1 + pad_h)
                crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))

                out = _extract_table_from_image_region(crop, debug=debug)
                out.update({"page": page_num, "method_origin": "pdf_text_search", "search_bbox_points": bbox, "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
                results_pages.append(out)
                summary["matches"] += 1
                summary["methods"].setdefault(out["method"], 0)
                summary["methods"][out["method"]] += 1

            summary["pages_found"] = len(results_pages)
            return {"pages": results_pages, "summary": summary}

    # --- If not found in vector text or ext != pdf: rasterize pages and OCR-search each page ---
    # Try convert all pages (or single image) and search with OCR word boxes
    pages_images = []
    if ext == "pdf":
        if convert_from_bytes is None or Image is None:
            logger.warning("pdf2image or Pillow missing; cannot perform PDF OCR fallback")
        else:
            try:
                pages_images = convert_from_bytes(file_bytes, dpi=dpi)
            except Exception:
                logger.exception("pdf2image full conversion failed; continuing with pdfplumber fallback")
                pages_images = []
    else:
        # try open as image
        if Image is None:
            logger.warning("Pillow missing; cannot open image")
        else:
            try:
                pages_images = [Image.open(BytesIO(file_bytes)).convert("RGB")]
            except Exception:
                logger.exception("Pillow failed to open uploaded file as image")
                pages_images = []

    # OCR-based page-by-page keyword search
    if pages_images:
        for idx, pil_page in enumerate(pages_images, start=1):
            matches = _find_keyword_in_image(pil_page, keywords)
            if not matches:
                continue
            # use first match
            (x, y, w, h), matched_kw = matches[0]
            pad_w = int(pil_page.width * 0.05)
            pad_h = int(pil_page.height * 0.18)
            crop_x0 = max(0, x - pad_w)
            crop_y0 = max(0, y - pad_h)
            crop_x1 = min(pil_page.width, x + w + pad_w)
            crop_y1 = min(pil_page.height, y + h + pad_h)
            crop = pil_page.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            out = _extract_table_from_image_region(crop, debug=debug)
            out.update({"page": idx, "method_origin": "ocr_page_search", "matched_keyword": matched_kw, "crop_pixels": (crop_x0, crop_y0, crop_x1, crop_y1)})
            results_pages.append(out)
            summary["matches"] += 1
            summary["methods"].setdefault(out["method"], 0)
            summary["methods"][out["method"]] += 1

        if results_pages:
            summary["pages_found"] = len(results_pages)
            return {"pages": results_pages, "summary": summary}

    # --- Try pdfplumber table extraction as last-resort on PDF text-layer tables ---
    if ext == "pdf" and pdfplumber is not None:
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        tables = []
                    for t in tables:
                        # check whether any cell contains any keyword (fuzzy)
                        found = False
                        for row in t:
                            for cell in row:
                                if cell and any(_fuzzy_contains(str(cell), kw, threshold=0.7) for kw in keywords):
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            df = pd.DataFrame(t).fillna("").applymap(_clean_text)
                            out = {"page": i, "method": "pdfplumber_table", "data": df.to_numpy().astype(str).tolist(), "html": df.to_html(index=False, header=False, border=1), "debug_image": None}
                            results_pages.append(out)
                            summary["matches"] += 1
                            summary["methods"].setdefault("pdfplumber_table", 0)
                            summary["methods"]["pdfplumber_table"] += 1
            if results_pages:
                summary["pages_found"] = len(results_pages)
                return {"pages": results_pages, "summary": summary}
        except Exception:
            logger.exception("pdfplumber fallback failed")

    # --- Final fallback: try full-page table detection on first page (image) or OCR fallback ---
    if pages_images:
        pil_page = pages_images[0]
        out = _extract_table_from_image_region(pil_page, debug=debug)
        out.update({"page": 1, "method_origin": "final_full_page_attempt"})
        results_pages.append(out)
        summary["methods"].setdefault(out["method"], 0)
        summary["methods"][out["method"]] += 1
        summary["pages_found"] = len(results_pages)
        return {"pages": results_pages, "summary": summary}

    # Nothing could be processed
    return {"pages": results_pages, "summary": {"error": "no page images available or required libs missing"}}

