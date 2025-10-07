from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import camelot
import io
import tempfile
import os

app = FastAPI(
    title="OCR + Table Extractor Web Tool",
    description="Extract text and tables from PDFs and images with high accuracy.",
    version="2.0.0"
)

# -------------------------------
# Allow CORS for any frontend (important for public access)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Root route (Render homepage)
# -------------------------------
@app.get("/")
async def root():
    return {
        "message": "✅ OCR + Table Extractor Backend is running successfully!",
        "usage": {
            "extract_text_and_tables": "POST /extract - Upload PDF or Image to extract both text and tables."
        }
    }


# -------------------------------
# Extraction Route (Text + Tables)
# -------------------------------
@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        # -------------------------------
        # CASE 1: PDF file
        # -------------------------------
        if filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(contents)
                tmp_pdf_path = tmp_pdf.name

            # Extract text via OCR
            images = convert_from_bytes(contents)
            extracted_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                extracted_text += f"\n--- Page {i+1} ---\n{text.strip()}\n"

            # Extract tables via Camelot
            tables_data = []
            try:
                tables = camelot.read_pdf(tmp_pdf_path, pages='all', flavor='stream')
                for t_index, table in enumerate(tables):
                    tables_data.append({
                        "page": t_index + 1,
                        "rows": table.df.values.tolist()
                    })
            except Exception as e:
                tables_data = [{"error": f"Table extraction failed: {str(e)}"}]

            # Cleanup temp
            os.unlink(tmp_pdf_path)

            return JSONResponse({
                "filename": file.filename,
                "extracted_text": extracted_text.strip(),
                "tables_extracted": len(tables_data),
                "table_data": tables_data
            })

        # -------------------------------
        # CASE 2: Image file
        # -------------------------------
        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = Image.open(io.BytesIO(contents))
            extracted_text = pytesseract.image_to_string(image)
            return JSONResponse({
                "filename": file.filename,
                "extracted_text": extracted_text.strip(),
                "tables_extracted": 0,
                "table_data": []
            })

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or image.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# -------------------------------
# Health check endpoint (Render)
# -------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}



