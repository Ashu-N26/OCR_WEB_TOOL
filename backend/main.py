# backend/main.py
"""
Backend FastAPI entrypoint for OCR Web Tool.

- Tries to import `extract_tables` from backend.table_extractor (wrapper) first,
  then backend.hybrid_extractor as a fallback.
- Exposes:
    GET  /health
    POST /extract?section=2.14&debug=false&dpi=300   (multipart file=@...)
- If a frontend/index.html (or frontend/build/index.html etc.) exists it will be served at "/".
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("backend.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
# Import extractor (robust)
# -------------------------
extract_tables = None
_import_error = None
try:
    # Prefer wrapper that delegates to hybrid extractor (if present)
    from backend.table_extractor import extract_tables as _extract  # type: ignore
    extract_tables = _extract
    logger.info("Loaded extractor from backend.table_extractor")
except Exception as e1:
    _import_error = e1
    try:
        from backend.hybrid_extractor import extract_tables as _extract2  # type: ignore
        extract_tables = _extract2
        logger.info("Loaded extractor from backend.hybrid_extractor")
    except Exception as e2:
        logger.warning("Could not import backend.table_extractor or backend.hybrid_extractor.")
        logger.debug("table_extractor import error: %s", e1)
        logger.debug("hybrid_extractor import error: %s", e2)
        _import_error = e2

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="OCR Web Tool Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health endpoint
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------
# Extract endpoint
# -------------------------
@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    section: Optional[str] = Query("2.14", description="Section keyword to search for (e.g. '2.14')"),
    debug: Optional[bool] = Query(False, description="Return debug overlay base64 in result"),
    dpi: Optional[int] = Query(300, description="DPI to rasterize PDF pages (higher = better OCR)"),
):
    """
    Upload a PDF or image and extract table(s) in the specified section.
    Returns: { pages: [...], summary: {...} } (structure produced by extract_tables)
    """
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    filename = file.filename or "file"
    try:
        contents = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file: %s", e)
        return JSONResponse({"error": f"failed to read uploaded file: {str(e)}"}, status_code=400)

    if not contents:
        return JSONResponse({"error": "empty file uploaded"}, status_code=400)

    if extract_tables is None:
        # Provide a helpful message including the import error for debugging
        msg = "Server misconfiguration: extractor module not available."
        logger.error(msg)
        logger.debug("Extractor import error: %s", _import_error)
        return JSONResponse({"error": msg, "import_error": repr(_import_error)}, status_code=500)

    # Optional: basic content type check (not required - extractor can try both)
    # allowed_types = ["application/pdf", "image/png", "image/jpeg", "image/tiff"]
    # if file.content_type and file.content_type not in allowed_types:
    #    logger.warning("Uploaded content-type %s not in allowed list", file.content_type)

    try:
        logger.info("Received file '%s' (%d bytes). section=%s debug=%s dpi=%s", filename, len(contents), section, debug, dpi)
        result = extract_tables(contents, filename, section_keyword=section, debug=debug, dpi=dpi)
        # Defensive: ensure result is serializable and shaped correctly
        if not isinstance(result, dict):
            logger.warning("Extractor returned non-dict result; wrapping it.")
            return JSONResponse({"pages": [], "summary": {"raw_result": str(result)}}, status_code=200)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Extraction failed: %s", e)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


# -------------------------
# Serve frontend if present (mount last so API wins)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (two levels up from backend/main.py)
candidates = [
    REPO_ROOT / "frontend" / "index.html",
    REPO_ROOT / "frontend" / "build" / "index.html",
    REPO_ROOT / "frontend" / "dist" / "index.html",
    REPO_ROOT / "index.html",
]

INDEX_PATH = None
for p in candidates:
    if p.exists():
        INDEX_PATH = p
        break

if INDEX_PATH:
    FRONTEND_DIR = INDEX_PATH.parent
    # Mount static files at root (API endpoints already defined above; they take precedence)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Serving frontend from: %s", FRONTEND_DIR)
else:
    # Root fallback that explains usage
    @app.get("/")
    async def root_no_frontend():
        return {
            "status": "running",
            "message": "OCR Web Tool backend running. No frontend found to serve at root.",
            "usage": {
                "health": "/health",
                "extract (POST multipart/form-data file)": "/extract?section=2.14&debug=false&dpi=300"
            }
        }


# -------------------------
# If run directly for local dev
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, log_level="info")
