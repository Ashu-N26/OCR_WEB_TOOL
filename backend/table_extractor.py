# backend/table_extractor.py
"""
Simple wrapper so existing imports from backend.table_extractor continue to work.
This just delegates to hybrid_extractor.extract_tables (if present).
"""

try:
    from backend.hybrid_extractor import extract_tables as extract_tables  # type: ignore
except Exception as e:
    # provide a graceful fallback function that reports the error
    def extract_tables(file_bytes, filename, section_keyword="2.14", extra_keywords=None, debug=False, dpi=300):
        return {
            "pages": [],
            "summary": {
                "error": "hybrid_extractor not importable",
                "exception": repr(e)
            }
        }
