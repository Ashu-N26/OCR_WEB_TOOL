from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io

app = FastAPI(
    title="OCR Web Tool",
    description="Extract text and tables from PDFs and Images accurately.",
    version="1.1.0"
)

# -------------------------------
# Allow CORS for all origins (important for frontend access)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Root route (Render health check / browser test)
# -------------------------------
@app.get("/")
async def root():
    return {
        "message": "✅ OCR Web Tool Backend is running successfully!",
        "usage": {
            "extract_text": "POST /extract - Upload a PDF or Image to extract text",
        }
    }


# -------------------------------
# OCR Extraction Route
# -------------------------------
@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        # Handle PDFs
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(contents)
            text_output = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                text_output += f"\n--- Page {i+1} ---\n{text.strip()}\n"
            return JSONResponse({"filename": file.filename, "extracted_text": text_output.strip()})

        # Handle Images (JPG, PNG, etc.)
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image)
            return JSONResponse({"filename": file.filename, "extracted_text": text.strip()})

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or image.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# -------------------------------
# Health check endpoint (for Render auto-check)
# -------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


