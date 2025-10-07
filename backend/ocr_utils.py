import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import camelot
import tempfile
import os

def preprocess_image(img: Image.Image) -> Image.Image:
    gray = img.convert('L')
    return gray

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        img = preprocess_image(img)
        text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_tables_from_pdf(pdf_path: str):
    tables = []
    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for t in camelot_tables:
            tables.append(t.df.to_dict(orient='records'))
    except Exception as e:
        print("Table extraction error:", e)
    return tables

