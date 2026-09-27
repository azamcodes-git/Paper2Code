"""
Paper2Code — FastAPI backend
Accepts a research paper PDF, runs the verification sandbox, generates an
executable Jupyter Notebook, and serves individual source files.

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
BASE_DIR   = Path(__file__).resolve().parent.parent   # repo root
SRC_DIR    = BASE_DIR / "src"
PAPERS_DIR = BASE_DIR / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

NOTEBOOK_PATH = SRC_DIR / "paper2code_pipeline.ipynb"

# Source files that the frontend can retrieve via /api/get-code
ALLOWED_SRC_FILES = {"model.py", "dataset.py", "verify_tensors.py"}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Paper2Code", version="2.0.0")


# ── root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    """Serve the React frontend."""
    html_path = SRC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(html_path), media_type="text/html")


# ── upload & process paper ────────────────────────────────────────────────────
@app.post("/api/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    """
    1. Save the uploaded PDF to papers/.
    2. Run verify_tensors.py in a subprocess.
    3. If verification passes, generate paper2code_pipeline.ipynb.
    4. Return logs, status, and whether the notebook is ready.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save the PDF
    dest = PAPERS_DIR / file.filename
    dest.write_bytes(await file.read())

    # Run verification sandbox
    verify_script = SRC_DIR / "verify_tensors.py"
    stdout = stderr = ""
    returncode = -1
    success = False
    try:
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR),
        )
        stdout    = result.stdout
        stderr    = result.stderr
        returncode = result.returncode
        success   = result.returncode == 0
    except subprocess.TimeoutExpired:
        stderr  = "Verification timed out after 120 s."
    except Exception as exc:
        stderr  = str(exc)

    # Generate notebook if verification passed
    notebook_ready = False
    notebook_error = ""
    if success:
        try:
            from src.notebook_generator import build_notebook
            build_notebook(SRC_DIR, NOTEBOOK_PATH)
            notebook_ready = True
        except Exception as exc:
            notebook_error = f"Notebook generation failed: {exc}"
            success = False          # surface the error

    return JSONResponse({
        "status": "ok" if success else "error",
        "filename": file.filename,
        "saved_to": str(dest.relative_to(BASE_DIR)),
        "verification": {
            "passed": returncode == 0,
            "stdout": stdout,
            "stderr": stderr + ("\n" + notebook_error if notebook_error else ""),
            "returncode": returncode,
        },
        "notebook_ready": notebook_ready,
        "message": (
            "Verification passed. Notebook ready — download via /api/download-notebook."
            if notebook_ready
            else "Paper saved but verification reported errors — see logs."
        ),
    })


# ── download notebook ─────────────────────────────────────────────────────────
@app.get("/api/download-notebook")
async def download_notebook():
    """
    Serve paper2code_pipeline.ipynb as a downloadable file attachment.
    Returns 404 if the notebook has not been generated yet (run upload first).
    """
    if not NOTEBOOK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Notebook not generated yet. Upload a paper first.",
        )
    return FileResponse(
        path=str(NOTEBOOK_PATH),
        media_type="application/octet-stream",
        filename="paper2code_pipeline.ipynb",
        headers={"Content-Disposition": 'attachment; filename="paper2code_pipeline.ipynb"'},
    )


# ── retrieve generated source files ──────────────────────────────────────────
@app.get("/api/get-code/{filename}")
async def get_code(filename: str):
    """Return the source code of model.py, dataset.py, or verify_tensors.py."""
    if filename not in ALLOWED_SRC_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not an allowed file. "
                   f"Choose from: {sorted(ALLOWED_SRC_FILES)}",
        )
    path = SRC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found in src/.")
    code_text = path.read_text(encoding="utf-8")
    return JSONResponse({
        "filename": filename,
        "content": code_text,   # legacy key
        "code": code_text,      # canonical key used by React UI
    })


# ── health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    notebook_exists = NOTEBOOK_PATH.exists()
    return {"status": "ok", "notebook_ready": notebook_exists}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
