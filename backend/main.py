import io
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hybrid_extractor import extract_table_from_pdf_or_image

# -----------------------------------------------------------
# App Initialization
# -----------------------------------------------------------
app = FastAPI(
    title="OCE Web Tool",
    description="An intelligent OCR + Table Extraction tool for aviation chart and document analysis",
    version="2.0.0"
)

# -----------------------------------------------------------
# Middleware (CORS)
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins — adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OCE_WebTool")

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.get("/")
def home():
    """
    Health check endpoint to verify if backend is running.
    """
    return {
        "status": "running",
        "message": "OCE Web Tool Backend Active and Ready"
    }


@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    """
    Upload an image or PDF and extract structured table data.
    Returns table in JSON + HTML format.
    """
    try:
        logger.info(f"File received: {file.filename}")

        # Read uploaded file bytes
        contents = await file.read()
        if not contents:
            logger.warning("Empty file uploaded.")
            return JSONResponse(
                content={"error": "Uploaded file is empty."},
                status_code=400
            )

        # Determine file type from extension
        filename = file.filename.lower()
        file_ext = filename.split(".")[-1]

        if file_ext not in ["pdf", "png", "jpg", "jpeg", "tif", "tiff"]:
            return JSONResponse(
                content={"error": f"Unsupported file type: {file_ext}. Upload PDF or image."},
                status_code=400
            )

        # Process using hybrid extraction (Tesseract + pdfplumber + OpenCV)
        logger.info("Starting hybrid extraction process...")
        result = extract_table_from_pdf_or_image(io.BytesIO(contents), file_ext)
        logger.info("Extraction complete.")

        # No table detected
        if not result or "data" not in result or not result["data"]:
            logger.warning("No tabular data extracted.")
            return JSONResponse(
                content={"message": "No tabular data found in the document."},
                status_code=200
            )

        # Return both structured JSON and HTML table
        return JSONResponse(
            content={
                "filename": file.filename,
                "rows": len(result["data"]),
                "columns": len(result["data"][0]) if result["data"] else 0,
                "data": result["data"],
                "html_table": result.get("html_table", ""),
                "message": "Table extracted successfully."
            },
            status_code=200
        )

    except Exception as e:
        logger.exception(f"Error extracting table: {e}")
        return JSONResponse(
            content={"error": f"Extraction failed: {str(e)}"},
            status_code=500
        )


# -----------------------------------------------------------
# Run (for local testing)
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
