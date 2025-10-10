# backend/main.py
"""
Robust FastAPI entrypoint for OCR_Web_Tool.

Behavior:
 - Exposes /health and /extract endpoints (API).
 - If frontend index.html exists (common paths), serve it at "/" and mount static assets.
 - Tries to import extract_tables from backend.table_extractor (preferred) or backend.hybrid_extractor.
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

# ---------------------------
# Attempt to import extractor (robust)
# ---------------------------
extract_tables = None
try:
    from backend.table_extractor import extract_tables as _ext  # wrapper
    extract_tables = _ext
    logger.info("Using backend.table_extractor")
except Exception:
    try:
        from backend.hybrid_extractor import extract_tables as _ext2
        extract_tables = _ext2
        logger.info("Using backend.hybrid_extractor")
    except Exception:
        extract_tables = None
        logger.warning("No extractor found: backend.table_extractor or backend.hybrid_extractor not importable")

# ---------------------------
# App init
# ---------------------------
app = FastAPI(title="OCR Web Tool - Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API routes
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    section: Optional[str] = Query("2.14", description="Section keyword to search for"),
    debug: Optional[bool] = Query(False, description="Return debug image base64"),
    dpi: Optional[int] = Query(300, description="DPI used to rasterize PDF pages"),
):
    """
    Upload a PDF or image and extract the table located at the provided section (default '2.14').
    Returns:
      { pages: [...], summary: {...} }
    """
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "empty file uploaded"}, status_code=400)

    if extract_tables is None:
        logger.error("Extractor module missing.")
        return JSONResponse({"error": "server misconfiguration: extractor not available"}, status_code=500)

    try:
        result = extract_tables(contents, file.filename or "file", section_keyword=section, debug=debug, dpi=dpi)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)

# ---------------------------
# Serve frontend (if present)
# ---------------------------
# Look for index.html in common frontend output locations:
REPO_ROOT = Path(__file__).resolve().parents[1]  # two levels up: repo root
candidates = [
    REPO_ROOT / "frontend" / "index.html",
    REPO_ROOT / "frontend" / "dist" / "index.html",
    REPO_ROOT / "frontend" / "build" / "index.html",
    REPO_ROOT / "frontend" / "public" / "index.html",
    REPO_ROOT / "index.html",
]

INDEX_PATH = None
for p in candidates:
    if p.exists():
        INDEX_PATH = p
        break

if INDEX_PATH:
    FRONTEND_DIR = INDEX_PATH.parent
    # Mount static files at / (after routes defined above so API endpoints take precedence)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving frontend index from: %s", INDEX_PATH)
else:
    # No frontend, provide helpful root endpoint
    @app.get("/", response_class=JSONResponse)
    async def root_no_frontend():
        return {
            "status": "running",
            "message": "OCR Web Tool backend running. No frontend served.",
            "endpoints": {
                "health": "/health",
                "extract (POST multipart/form-data file)": "/extract"
            }
        }

# If you still want an HTML fallback even when no frontend present, uncomment below:
# @app.get("/", response_class=HTMLResponse)
# async def root_html_fallback():
#     return "<html><body><h1>OCR Web Tool</h1><p>Use /extract endpoint to upload files.</p></body></html>"




