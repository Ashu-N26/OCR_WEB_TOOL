# backend/main.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import traceback
import os

# package-aware import
from backend.hybrid_extractor import extract_tables

app = FastAPI(title="OCR Web Tool - Hybrid Extractor")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...), debug: bool = Query(False), dpi: int = Query(300)):
    """
    Upload PDF or image file. Query params:
      - debug (bool): if true, response includes debug_image (base64) for visualization
      - dpi (int): when converting PDF pages to images via pdf2image (default 300)
    Response: { pages: [ {page, method, html, data, debug_image, boxes, cols, rows} ], summary: {...} }
    """
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)
    filename = file.filename or "file"
    contents = await file.read()
    try:
        result = extract_tables(contents, filename, debug=debug, dpi=dpi)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Extraction failed: %s", e)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


# Serve frontend (if present)
from fastapi.staticfiles import StaticFiles
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.info("Frontend folder not found; static serving disabled.")









