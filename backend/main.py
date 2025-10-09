# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import os
import traceback

import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes

# package-aware import (backend.table_extractor)
from backend.table_extractor import extract_tables_from_pdf_bytes, extract_tables_from_image

app = FastAPI(title="OCR Web Tool", description="Extract text and tables from PDFs/images")

# Allow CORS for all origins (public link)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def _ocr_text_from_image_bytes(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception:
        return ""

def _text_from_pdf_native(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                texts.append(txt)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _text_from_pdf_ocr(pdf_bytes: bytes) -> str:
    text = ""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        for i, p in enumerate(pages, start=1):
            page_text = pytesseract.image_to_string(p, lang="eng")
            text += f"\n--- Page {i} ---\n{page_text.strip()}\n"
    except Exception:
        pass
    return text.strip()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Single-file upload handler.
    Returns JSON: { filename, extracted_text, tables: [ {page, type, data}, ... ] }
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename = file.filename or "file"
    contents = await file.read()

    try:
        if filename.lower().endswith(".pdf"):
            # tables
            tables = extract_tables_from_pdf_bytes(contents)
            # text: try native first, then OCR fallback
            text = _text_from_pdf_native(contents)
            if not text or len(text) < 20:
                text = _text_from_pdf_ocr(contents)
            return JSONResponse({"filename": filename, "extracted_text": text, "tables": tables})

        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            text = _ocr_text_from_image_bytes(contents)
            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                tables = extract_tables_from_image(img)
            except Exception:
                tables = []
            return JSONResponse({"filename": filename, "extracted_text": text, "tables": tables})

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or image.")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}\n{tb}")


# Serve frontend static files (combined single deploy)
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print("⚠️ Frontend folder not found; static serving disabled.")






