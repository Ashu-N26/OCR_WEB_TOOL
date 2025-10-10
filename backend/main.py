# backend/main.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging, traceback, os

# try import table extractor from wrapper names (resilient)
_extract = None
try:
    # preferred wrapper (keeps old import paths working)
    from backend.table_extractor import extract_tables as _extract
except Exception:
    try:
        from backend.hybrid_extractor import extract_tables as _extract
    except Exception:
        _extract = None

app = FastAPI(title="OCE Web Tool - Backend")
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
    section: str = Query("2.14", description="Section keyword to search for"),
    debug: bool = Query(False, description="Return debug overlay as base64"),
    dpi: int = Query(300, description="DPI used to rasterize PDF pages")
):
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "empty file uploaded"}, status_code=400)

    if _extract is None:
        logger.error("No extractor module available (backend.table_extractor or backend.hybrid_extractor)")
        return JSONResponse({"error": "Server misconfiguration: extractor not installed"}, status_code=500)

    try:
        # _extract expects: (file_bytes, filename, section_keyword=..., debug=..., dpi=...)
        result = _extract(contents, file.filename or "file", section_keyword=section, debug=debug, dpi=dpi)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Extraction failed: %s", e)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)



