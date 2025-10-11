"""
backend/table_extractor.py

Wrapper around the hybrid extractor. This file:
 - tries to import and use backend.hybrid_extractor.extract_tables
 - provides a stable public API used by main.py variants:
     - extract_tables_from_pdf_bytes(...)
     - extract_tables_from_image(...)
     - extract_tables_auto(...)
 - falls back to a conservative OCR-only extraction if hybrid_extractor is missing or fails.

Return format (consistent):
    {
      "status": "success",
      "result": <dict returned by hybrid extractor or fallback>
    }
    or
    {
      "status": "error",
      "message": "short error string",
      "details": <exception or debug info>
    }

Notes:
 - section_keyword defaults to "2.14" (you can change per-call).
 - debug=True will include extra logs in the returned "details" field where possible.
"""

from typing import Optional, Dict, Any, Tuple
import io
import logging
import imghdr

# Optional dependencies - imported lazily where needed
try:
    from PIL import Image
except Exception:
    Image = None

# Try to import the hybrid extractor (relative & absolute attempts)
hybrid_extract = None
try:
    # prefer package-style import if backend package is installed
    from backend.hybrid_extractor import extract_tables as hybrid_extract
except Exception:
    try:
        # fallback to relative import (when running from repository root)
        from .hybrid_extractor import extract_tables as hybrid_extract
    except Exception:
        hybrid_extract = None

# Setup logger
logger = logging.getLogger("table_extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------
# Util: detect file type
# -----------------------------
def _is_pdf_bytes(b: bytes) -> bool:
    if not b:
        return False
    # PDF files start with "%PDF"
    return b[:4] == b"%PDF"


def _guess_image_extension(b: bytes) -> str:
    # Use imghdr to guess between png/jpeg/gif. Return ext without dot.
    try:
        t = imghdr.what(None, h=b)
        if t:
            if t == "jpeg":
                return "jpg"
            return t
    except Exception:
        pass
    # fallback: try PIL to detect format
    if Image is not None:
        try:
            img = Image.open(io.BytesIO(b))
            fmt = img.format or ""
            img.close()
            if fmt:
                return fmt.lower()
        except Exception:
            pass
    return "png"


# -----------------------------
# Fallback OCR-only extractor
# -----------------------------
def _ocr_text_only_from_bytes(file_bytes: bytes, filename: str = "file", dpi: int = 300, debug: bool = False) -> Dict[str, Any]:
    """
    Conservative fallback: convert bytes to image(s) and run pytesseract image_to_string
    Returns dict with pages (page_num -> text) and a short summary.
    """
    out = {"pages": [], "summary": {"method": "ocr_only_fallback", "pages_processed": 0}}
    try:
        # If PDF, render pages via PyMuPDF/pdf2image if available, else try PyMuPDF then convert_from_bytes
        pages = []
        if _is_pdf_bytes(file_bytes):
            # try PyMuPDF first (fitz)
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=dpi)
                    pages.append(pix.tobytes("png"))
                try:
                    doc.close()
                except Exception:
                    pass
            except Exception:
                # fallback to pdf2image if installed
                try:
                    from pdf2image import convert_from_bytes
                    pil_pages = convert_from_bytes(file_bytes, dpi=dpi)
                    for p in pil_pages:
                        buf = io.BytesIO()
                        p.save(buf, format="PNG")
                        pages.append(buf.getvalue())
                except Exception:
                    logger.exception("OCR fallback: unable to rasterize PDF (no fitz/pdf2image).")
                    return {"error": "cannot rasterize pdf for OCR fallback; missing dependencies."}
        else:
            # treat as image
            pages = [file_bytes]

        # run tesseract on each image page
        try:
            import pytesseract
        except Exception:
            return {"error": "pytesseract not available for OCR fallback."}

        for idx, page_bytes in enumerate(pages, start=1):
            try:
                if Image is None:
                    # try to import Pillow on demand
                    from PIL import Image as PILImage
                    img = PILImage.open(io.BytesIO(page_bytes)).convert("RGB")
                else:
                    img = Image.open(io.BytesIO(page_bytes)).convert("RGB")
                text = pytesseract.image_to_string(img)
                out["pages"].append({"page": idx, "text": text})
                out["summary"]["pages_processed"] += 1
            except Exception as e:
                logger.exception("OCR fallback: page processing failed for page %s: %s", idx, e)
                out["pages"].append({"page": idx, "error": str(e)})
        return out
    except Exception as e:
        logger.exception("OCR fallback failed: %s", e)
        return {"error": "ocr_fallback_exception", "details": str(e)}


# -----------------------------
# Public wrapper functions
# -----------------------------
def extract_tables_from_pdf_bytes(pdf_bytes: bytes,
                                  filename: str = "file.pdf",
                                  section_keyword: Optional[str] = "2.14",
                                  debug: bool = False,
                                  dpi: int = 300) -> Dict[str, Any]:
    """
    Primary entry for PDF bytes. Tries hybrid_extractor.extract_tables if available.
    Returns standardized dict with status/result or error.
    """
    if not pdf_bytes:
        return {"status": "error", "message": "empty file bytes"}

    if hybrid_extract is not None:
        try:
            result = hybrid_extract(file_bytes=pdf_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
            return {"status": "success", "result": result}
        except TypeError:
            # some earlier hybrid implementations used different parameter ordering/names
            try:
                result = hybrid_extract(pdf_bytes, filename, section_keyword, debug, dpi)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.exception("hybrid_extract raised (try-2): %s", e)
        except Exception as e:
            logger.exception("hybrid_extract failed: %s", e)

    # fallback: OCR only
    if debug:
        logger.info("hybrid_extractor not available or failed -> falling back to OCR-only extraction")
    fallback = _ocr_text_only_from_bytes(pdf_bytes, filename=filename, dpi=dpi, debug=debug)
    return {"status": "success", "result": {"engine": "ocr_fallback", "data": fallback}}


def extract_tables_from_image(image_bytes: bytes,
                              filename: str = "image.png",
                              section_keyword: Optional[str] = "2.14",
                              debug: bool = False,
                              dpi: int = 300) -> Dict[str, Any]:
    """
    Primary entry for image bytes (PNG/JPG/TIFF). For compatibility, delegates to hybrid extractor (which
    inspects extension) or falls back to OCR-only routine.
    """
    if not image_bytes:
        return {"status": "error", "message": "empty image bytes"}

    # Ensure filename has an image extension
    if "." not in filename:
        ext = _guess_image_extension(image_bytes)
        filename = f"{filename}.{ext}"

    if hybrid_extract is not None:
        try:
            result = hybrid_extract(file_bytes=image_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.exception("hybrid_extractor failed for image: %s", e)

    # fallback OCR
    fallback = _ocr_text_only_from_bytes(image_bytes, filename=filename, dpi=dpi, debug=debug)
    return {"status": "success", "result": {"engine": "ocr_fallback_image", "data": fallback}}


def extract_tables_auto(file_bytes: bytes,
                        filename: Optional[str] = None,
                        section_keyword: Optional[str] = "2.14",
                        debug: bool = False,
                        dpi: int = 300) -> Dict[str, Any]:
    """
    Auto-detect file type and call appropriate function above.
    If filename missing, attempt to guess from header bytes.
    """
    if not file_bytes:
        return {"status": "error", "message": "empty file bytes"}

    if not filename:
        if _is_pdf_bytes(file_bytes):
            filename = "file.pdf"
        else:
            ext = _guess_image_extension(file_bytes)
            filename = f"file.{ext}"

    # route by extension
    ext = filename.split(".")[-1].lower()
    try:
        if ext == "pdf":
            return extract_tables_from_pdf_bytes(file_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp", "gif"):
            return extract_tables_from_image(file_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
        else:
            # unknown extension: try pdf path first if header says pdf, else image fallback
            if _is_pdf_bytes(file_bytes):
                return extract_tables_from_pdf_bytes(file_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
            else:
                return extract_tables_from_image(file_bytes, filename=filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
    except Exception as e:
        logger.exception("extract_tables_auto error: %s", e)
        return {"status": "error", "message": "extract_tables_auto_exception", "details": str(e)}


# back-compat names some main.py variants might import:
extract_tables_from_bytes = extract_tables_from_pdf_bytes
extract_tables = extract_tables_auto


# -----------------------------
# When run directly, quick self-test (not for production)
# -----------------------------
if __name__ == "__main__":
    # quick smoke test: try to import hybrid extractor and call with no bytes -> expect graceful error
    print("table_extractor wrapper loaded. hybrid_extract available:", hybrid_extract is not None)

