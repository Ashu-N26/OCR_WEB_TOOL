# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import traceback
import logging

# package-aware import
from backend.table_extractor import extract_tables

app = FastAPI(title="OCR Web Tool (Text + Table Extractor)")

# Logging
logging.basicConfig(level=logging.INFO)

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
async def extract(file: UploadFile = File(...), debug: bool = Query(False)):
    """
    Upload a PDF or Image and extract text + tables.
    Query param debug=true returns debug_image (base64) in response for visualization.
    """
    if not file:
        return JSONResponse({"error": "no file uploaded"}, status_code=400)

    filename = file.filename or "file"
    contents = await file.read()

    try:
        result = extract_tables(contents, filename, debug=debug)
        return JSONResponse(result)
    except Exception as e:
        tb = traceback.format_exc()
        logging.exception("Extraction failed")
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)








