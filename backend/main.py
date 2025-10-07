from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ocr_utils import extract_text_from_image, extract_text_from_pdf, extract_tables_from_pdf
import shutil, tempfile, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        if suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
            text = extract_text_from_image(tmp_path)
            tables = []
        elif suffix == ".pdf":
            text = extract_text_from_pdf(tmp_path)
            tables = extract_tables_from_pdf(tmp_path)
        else:
            text = ""
            tables = []

        results.append({
            "filename": file.filename,
            "text": text,
            "tables": tables
        })
        os.remove(tmp_path)
    return {"results": results}
