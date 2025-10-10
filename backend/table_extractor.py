# backend/table_extractor.py
"""
Wrapper so backend.main can always import a known function `extract_tables`.
If hybrid_extractor exists, use it. Otherwise, provide a simple fallback.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger("table_extractor")

# Try to import the hybrid extractor (your advanced implementation)
try:
    from backend.hybrid_extractor import extract_tables as _hybrid_extract
except Exception:
    _hybrid_extract = None
    logger.info("backend.hybrid_extractor not found; fallback extractor will be used.")


def _fallback_extract(file_bytes: bytes, filename: str, section_keyword: str = "2.14", debug: bool = False, dpi: int = 300) -> Dict[str, Any]:
    # Very small fallback: returns raw OCR text lines as single-column table
    try:
        from io import BytesIO
        from PIL import Image
        import pytesseract
        from pdf2image import convert_from_bytes
    except Exception as e:
        return {"pages": [], "summary": {"error": f"fallback dependencies missing: {e}"}}

    pages = []
    if filename.lower().endswith(".pdf"):
        try:
            pil_pages = convert_from_bytes(file_bytes, dpi=dpi)
            images = [p.convert("RGB") for p in pil_pages]
        except Exception as e:
            return {"pages": [], "summary": {"error": f"pdf2image failed: {e}"}}
    else:
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            images = [img]
        except Exception as e:
            return {"pages": [], "summary": {"error": f"image open failed: {e}"}}

    out_pages = []
    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img)
        rows = [r.strip() for r in text.splitlines() if r.strip()]
        out_pages.append({"page": i, "method": "fallback_ocr", "data": [{"line": r} for r in rows], "html": "<pre>{}</pre>".format("\n".join(rows))})
    return {"pages": out_pages, "summary": {"pages": len(out_pages), "method": "fallback"}}


def extract_tables(file_bytes: bytes, filename: str, section_keyword: str = "2.14", debug: bool = False, dpi: int = 300):
    if _hybrid_extract:
        return _hybrid_extract(file_bytes, filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
    return _fallback_extract(file_bytes, filename, section_keyword=section_keyword, debug=debug, dpi=dpi)
