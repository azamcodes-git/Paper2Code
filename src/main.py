"""
Paper2Code — FastAPI backend
Accepts a research paper PDF, runs the verification sandbox, and returns
generated PyTorch source files (model.py, dataset.py).

Start with:
    python src/main.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so that `src.main` is importable
# both in the main process and in uvicorn's reload subprocess.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
SRC_DIR  = BASE_DIR / "src"
PAPERS_DIR = BASE_DIR / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

# Source files that the frontend can retrieve
ALLOWED_SRC_FILES = {"model.py", "dataset.py", "verify_tensors.py"}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Paper2Code", version="1.0.0")


# ── root ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    """Serve the minimalist frontend page."""
    html_path = SRC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(html_path), media_type="text/html")


# ── upload & process paper ──────────────────────────────────────────────────
@app.post("/api/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    """
    Accept a PDF upload, save it to papers/, run verify_tensors.py,
    and return logs + status.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save the uploaded file
    dest = PAPERS_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    # Run the verification sandbox
    verify_script = SRC_DIR / "verify_tensors.py"
    try:
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BASE_DIR),
        )
        stdout = result.stdout
        stderr = result.stderr
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Verification timed out after 120 s."
        success = False
    except Exception as exc:
        stdout = ""
        stderr = str(exc)
        success = False

    return JSONResponse({
        "status": "ok" if success else "error",
        "filename": file.filename,
        "saved_to": str(dest.relative_to(BASE_DIR)),
        "verification": {
            "passed": success,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode if "result" in dir() else -1,
        },
        "message": (
            "Paper saved and verification passed. Retrieve code via /api/get-code/{filename}."
            if success
            else "Paper saved but verification reported errors — see logs."
        ),
    })


# ── retrieve generated source files ─────────────────────────────────────────
@app.get("/api/get-code/{filename}")
async def get_code(filename: str):
    """
    Return the source code of model.py, dataset.py, or verify_tensors.py.
    """
    if filename not in ALLOWED_SRC_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not an allowed file. "
                   f"Choose from: {sorted(ALLOWED_SRC_FILES)}",
        )
    path = SRC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found in src/.")
    return JSONResponse({
        "filename": filename,
        "content": path.read_text(encoding="utf-8"),
    })


# ── health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Pass the app object directly — avoids the reload-subprocess sys.path issue.
    uvicorn.run(app, host="0.0.0.0", port=port)
