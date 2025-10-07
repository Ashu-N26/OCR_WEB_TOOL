# install OS-level dependencies required by tesseract, poppler, camelot, opencv etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre \                      # required by tabula-py (if using)
    pkg-config \
    poppler-utils \                     # pdf2image / general PDF utilities
    libpoppler-cpp-dev \
    tesseract-ocr \                     # tesseract binary
    tesseract-ocr-eng \                 # english language pack (optional, safe)
    libtesseract-dev \
    libleptonica-dev \
    ghostscript \                       # used by Camelot and PDF workflows
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \                   # OpenGL lib for headless opencv
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


