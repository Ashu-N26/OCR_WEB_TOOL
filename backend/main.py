# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import traceback
import logging

from backend.hybrid_extractor import extract_tables

app = FastAPI(title="OCR Web Tool")

# serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

logger = logging.getLogger("ocr_web_tool")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    section: str = Form(None),
    debug: bool = Form(False),
    dpi: int = Form(300),
):
    """
    Endpoint receives multipart/form-data:
      - file: the uploaded PDF/image
      - section: optional section keyword (if provided we try to find that section first)
      - debug: optional boolean to include debug overlays and extra notes
      - dpi: rasterization DPI for scanned PDFs
    """

    content = await file.read()
    filename = file.filename or "uploaded"

    # --- KEY CHANGE: if section is empty or None, we pass section_keyword=None so extractor will do a full-scan fallback ---
    section_keyword = section.strip() if section and section.strip() else None

    try:
        # enable debug=True to get overlay images and more notes during testing
        result = extract_tables(content, filename=filename, section_keyword=section_keyword, debug=bool(debug), dpi=int(dpi))
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Extraction failed: %s", e)
        return JSONResponse({"error": "extraction_failed", "message": str(e), "trace": tb}, status_code=500)
