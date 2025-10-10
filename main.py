# main.py (repo root)
"""
Root entry so 'uvicorn main:app' works reliably in Render/Docker.
It simply imports the FastAPI `app` from backend.main so the package layout
can remain under the 'backend' folder.
"""

try:
    # Prefer the packaged backend app
    from backend.main import app  # type: ignore
except Exception:
    # If something is misconfigured, create a minimal fallback app so the import error is visible
    from fastapi import FastAPI
    import logging
    logging.exception("Could not import backend.main; using fallback app.")
    app = FastAPI(title="Fallback app - backend.main import failed")

if __name__ == "__main__":
    # Run directly for local testing
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
