from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDFs
import io
import os

app = FastAPI(title="OCR Web Tool", description="Extract text/data from images and PDFs with high accuracy")

# Allow frontend requests (safe even if frontend served from same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- OCR Utility Functions -------- #
def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")

# -------- API Routes -------- #
@app.get("/health")
def health_check():
    """Simple endpoint for Render health monitoring"""
    return {"status": "OK"}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded image or PDF.
    Returns structured JSON response.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(content)
    elif filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or image.")

    if not text.strip():
        return JSONResponse({"text": "", "message": "No readable text found in the uploaded file"})

    return {"filename": filename, "extracted_text": text}

# -------- Serve Frontend -------- #
# Serve HTML/JS/CSS from the frontend folder
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print("⚠️ Frontend folder not found — static file serving disabled.")

# -------- Run Locally (optional) -------- #
# Only used for local dev — ignored by Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)




