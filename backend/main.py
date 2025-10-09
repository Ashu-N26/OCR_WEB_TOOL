"""
backend/main.py

FastAPI app to accept file uploads and extract tables from a specified section (default "2.14").
Expose endpoint: POST /extract?section=2.14&debug=true&dpi=300
"""

import io
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.hybrid_extractor import extract_tables  # function defined above

app = FastAPI(title="OCE Web Tool - Hybrid Extractor")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

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
async def extract_endpoint(
    file: UploadFile = File(...),
    section: Optional[str] = Query("2.14", description="Section keyword to search for, e.g. 2.14"),
    debug: Optional[bool] = Query(False, description="If true, returns debug_image base64"),
    dpi: Optional[int] = Query(300, description="DPI for PDF->image conversion")
):
    """
    Upload a PDF or image and extract the table located at the provided section (default '2.14').
    Returns:
      {
        pages: [
          { page, method, data (list of row dicts), html, debug_image (base64) }
        ],
        summary: {...}
      }
    """
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    filename = file.filename or "file"
    contents = await file.read()

    # Basic validation
    if len(contents) == 0:
        return JSONResponse({"error": "empty file uploaded"}, status_code=400)

    try:
        result = extract_tables(contents, filename, section_keyword=section, debug=debug, dpi=dpi)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Extraction error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

