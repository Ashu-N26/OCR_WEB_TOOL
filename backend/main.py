# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import os
import tempfile
import traceback

import pytesseract
from PIL import Image

import fitz  # PyMuPDF for native PDF text
from pdf2image import convert_from_bytes

# local table extractor module
from table_extractor import extract_tables_from_pdf_bytes, extract_tables_from_image

app = FastAPI(title="OCR Web Tool (Text + Table Extractor)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _extract_text_from_image_bytes(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception:
        return ""


def _extract_text_from_pdf_bytes_native(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                text_parts.append(txt)
        out = "\n".join(text_parts).strip()
        return out
    except Exception:
        return ""


def _extract_text_from_pdf_bytes_ocr(pdf_bytes: bytes) -> str:
    text = ""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang="eng")
            text += f"\n--- Page {i+1} ---\n{page_text.strip()}\n"
    except Exception:
        pass
    return text.strip()


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Extract text and tables from uploaded PDF or image.
    Response:
    {
      "filename": "...",
      "extracted_text": "...",
      "tables": [ { "page": n, "type": "...", "data": [ [row], [row], ... ] }, ... ]
    }
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename = file.filename or "file"
    contents = await file.read()

    try:
        if filename.lower().endswith(".pdf"):
            # 1) Extract tables (Camelot or OCR fallback)
            tables = extract_tables_from_pdf_bytes(contents)

            # 2) Try native PDF text extraction first
            text = _extract_text_from_pdf_bytes_native(contents)
            if not text or len(text) < 30:
                # fallback to OCR raster pages
                text = _extract_text_from_pdf_bytes_ocr(contents)

            return JSONResponse({
                "filename": filename,
                "extracted_text": text,
                "tables": tables
            })

        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # image file
            # text via pytesseract
            text = _extract_text_from_image_bytes(contents)
            # tables from image
            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                tables = extract_tables_from_image(img)
            except Exception:
                tables = []

            return JSONResponse({
                "filename": filename,
                "extracted_text": text,
                "tables": tables
            })
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or image.")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}\n{tb}")


# Serve frontend static files (combined deployment)
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print("Frontend folder not found; static serving disabled.")





