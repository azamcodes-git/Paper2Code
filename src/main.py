"""
Paper2Code — FastAPI backend
Accepts a research paper PDF, calls IBM Bob (watsonx) to generate paper-specific
PyTorch code, runs tensor verification, builds a Jupyter Notebook, and serves the
generated files back to the React frontend.

Start with:
    py src/main.py
"""

import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Repo root on sys.path so `src.*` imports work from any cwd
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
SRC_DIR     = BASE_DIR / "src"
PAPERS_DIR  = BASE_DIR / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

NOTEBOOK_PATH     = SRC_DIR / "paper2code_pipeline.ipynb"
ALLOWED_SRC_FILES = {"model.py", "dataset.py", "verify_tensors.py"}

# ---------------------------------------------------------------------------
# IBM watsonx / Bob credentials  (loaded from .env — never hardcoded)
# ---------------------------------------------------------------------------
WATSONX_API_KEY = os.getenv("IBM_CLOUD_API_KEY", "")
WATSONX_URL     = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_MODEL   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")

# ---------------------------------------------------------------------------
# FastAPI app + CORS (allows browser on any port to reach the API)
# ---------------------------------------------------------------------------
app = FastAPI(title="Paper2Code", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def _extract_pdf_text(pdf_path: Path, max_chars: int = 12_000) -> str:
    """
    Try pypdf first, then pdfminer.six; fall back to a filename stub if neither
    is installed.  Returns at most max_chars of plain text.
    """
    text = ""
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
            if len(text) >= max_chars:
                break
    except Exception:
        try:
            from pdfminer.high_level import extract_text as _pm  # type: ignore
            text = _pm(str(pdf_path))
        except Exception:
            text = f"[PDF text extraction unavailable — file: {pdf_path.name}]"
    return text[:max_chars].strip()


# ---------------------------------------------------------------------------
# IBM watsonx.ai REST helper
# ---------------------------------------------------------------------------
def _iam_token() -> str:
    """Exchange IBM_CLOUD_API_KEY for a short-lived IAM bearer token."""
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey":     WATSONX_API_KEY,
    }).encode()
    req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())["access_token"]
    except Exception as exc:
        raise RuntimeError(f"IAM token fetch failed: {exc}") from exc


def _watsonx_generate(prompt: str) -> str:
    """
    POST to /ml/v1/text/generation and return the generated text.
    Raises RuntimeError on any failure.
    """
    token = _iam_token()
    gen_url = (
        WATSONX_URL.rstrip("/") + "/ml/v1/text/generation?version=2024-05-01"
    )
    payload = json.dumps({
        "model_id":   WATSONX_MODEL,
        "project_id": WATSONX_PROJECT,
        "input":      prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens":  3000,
        },
    }).encode()
    req = urllib.request.Request(
        gen_url,
        data=payload,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["results"][0]["generated_text"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"watsonx HTTP {exc.code}: {body[:400]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"watsonx generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# IBM Bob prompts
# ---------------------------------------------------------------------------
_MODEL_PROMPT = textwrap.dedent("""\
    You are IBM Bob 2.0, an expert AI that converts research papers into
    production-quality PyTorch code.

    Read the paper text below, then output a COMPLETE Python file (model.py)
    that implements every neural-network architecture in the paper.

    Rules:
    - Module docstring citing paper title, venue, and DOI.
    - NUM_CLASSES, NUM_BANDS (or equivalent input size), DROPOUT as constants.
    - One nn.Module subclass per architecture.
    - Each forward(x) accepts (N, input_size) float32 and returns (N, NUM_CLASSES).
    - build_model(name) factory.
    - Only torch / torch.nn imports — no extras.
    - Output RAW Python only — no markdown, no fences, no commentary outside code.

    Paper text:
    {paper_text}
""")

_DATASET_PROMPT = textwrap.dedent("""\
    You are IBM Bob 2.0, an expert AI that converts research papers into
    production-quality PyTorch code.

    Read the paper text below, then output a COMPLETE Python file (dataset.py)
    implementing the Dataset & DataLoader pipeline described in the paper.

    Rules:
    - Module docstring citing the paper.
    - torch.utils.data.Dataset subclass for the paper's data format.
    - CSV/numpy loader matching the paper's data layout.
    - StandardScaler-style normaliser fitted on training data only.
    - make_dataloaders(features, labels) -> (train_loader, val_loader).
    - Only torch, numpy, standard library (pandas as optional guarded import).
    - Output RAW Python only — no markdown, no fences.

    Paper text:
    {paper_text}
""")

_VERIFY_PROMPT = textwrap.dedent("""\
    You are IBM Bob 2.0, an expert AI that converts research papers into
    production-quality PyTorch code.

    Read the paper text below, then output a COMPLETE Python verification script
    (verify_tensors.py) that:
    1. Imports every model class from src.model.
    2. Instantiates each model.
    3. Creates a synthetic input tensor matching the paper's input shape.
    4. Runs a forward pass, asserts output shape == (BATCH, NUM_CLASSES).
    5. Prints [PASS] or [FAIL] for every model.
    6. sys.exit(1) if any check fails, sys.exit(0) on full success.

    Only torch, numpy, standard library.
    Output RAW Python only — no markdown, no fences.

    Paper text:
    {paper_text}
""")


def _generate_code_for_paper(paper_text: str) -> dict:
    """
    Call IBM Bob (watsonx) to generate three source files for the given paper.
    Returns {'model.py': ..., 'dataset.py': ..., 'verify_tensors.py': ...}.
    Raises RuntimeError if credentials are absent or API calls fail.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT:
        raise RuntimeError(
            "IBM watsonx credentials not configured. "
            "Add IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID to your .env file."
        )

    return {
        "model.py":          _watsonx_generate(_MODEL_PROMPT.format(paper_text=paper_text)),
        "dataset.py":        _watsonx_generate(_DATASET_PROMPT.format(paper_text=paper_text)),
        "verify_tensors.py": _watsonx_generate(_VERIFY_PROMPT.format(paper_text=paper_text)),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    """Serve the React SPA."""
    html = SRC_DIR / "index.html"
    if not html.exists():
        raise HTTPException(404, "index.html not found")
    return FileResponse(str(html), media_type="text/html")


@app.post("/api/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    """
    Full Paper2Code pipeline
    ────────────────────────
    1  Save uploaded PDF to papers/
    2  Extract plain text from the PDF
    3  Call IBM Bob 2.0 (watsonx) → generate model.py / dataset.py / verify_tensors.py
       • If credentials are absent, fall back to the existing src/ files
    4  Run verify_tensors.py in a subprocess to validate tensor shapes
    5  Build paper2code_pipeline.ipynb from the (possibly new) src/ files
    6  Return structured JSON with logs, verification output, and notebook status
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # 1 ── Save PDF ─────────────────────────────────────────────────────────
    dest = PAPERS_DIR / file.filename
    dest.write_bytes(await file.read())

    pipeline_logs: list[str] = []
    gen_error = ""
    generated: dict[str, str] = {}

    # 2 ── Extract PDF text ─────────────────────────────────────────────────
    pipeline_logs.append("  [INFO] Extracting text from PDF…")
    paper_text = _extract_pdf_text(dest)
    pipeline_logs.append(
        f"  [INFO] Extracted {len(paper_text):,} chars from {file.filename}"
    )

    # 3 ── IBM Bob code generation ──────────────────────────────────────────
    pipeline_logs.append(
        "  [INFO] Calling IBM Bob 2.0 (watsonx) — generating source files…"
    )
    try:
        generated = _generate_code_for_paper(paper_text)
        for fname, src_code in generated.items():
            (SRC_DIR / fname).write_text(src_code, encoding="utf-8")
            pipeline_logs.append(
                f"  [PASS] IBM Bob wrote {fname} "
                f"({len(src_code.splitlines())} lines)"
            )
    except RuntimeError as exc:
        gen_error = str(exc)
        pipeline_logs.append(f"  [WARN] IBM Bob unavailable → {gen_error}")
        pipeline_logs.append(
            "  [INFO] Falling back to existing src/ files for verification."
        )

    # 4 ── Tensor verification ──────────────────────────────────────────────
    pipeline_logs.append("  [INFO] Running tensor verification sandbox…")
    verify_script = SRC_DIR / "verify_tensors.py"
    v_stdout = v_stderr = ""
    returncode = -1
    success = False
    try:
        result = subprocess.run(
            [sys.executable, str(verify_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BASE_DIR),
        )
        v_stdout   = result.stdout
        v_stderr   = result.stderr
        returncode = result.returncode
        success    = result.returncode == 0
    except subprocess.TimeoutExpired:
        v_stderr = "Verification timed out after 120 s."
    except Exception as exc:
        v_stderr = str(exc)

    # 5 ── Build notebook ───────────────────────────────────────────────────
    notebook_ready = False
    notebook_error = ""
    if success:
        try:
            from src.notebook_generator import build_notebook  # noqa: PLC0415
            build_notebook(SRC_DIR, NOTEBOOK_PATH)
            notebook_ready = True
            pipeline_logs.append(
                "  [PASS] Notebook built → paper2code_pipeline.ipynb"
            )
        except Exception as exc:
            notebook_error = f"Notebook generation failed: {exc}"
            success = False
            pipeline_logs.append(f"  [FAIL] {notebook_error}")

    # 6 ── Assemble response ────────────────────────────────────────────────
    full_stdout = "\n".join(pipeline_logs) + ("\n" + v_stdout if v_stdout else "")
    full_stderr = v_stderr + ("\n" + notebook_error if notebook_error else "")

    return JSONResponse({
        "status":   "ok" if success else "error",
        "filename": file.filename,
        "saved_to": str(dest.relative_to(BASE_DIR)),
        "ibm_bob": {
            "credentials_present": bool(WATSONX_API_KEY and WATSONX_PROJECT),
            "files_generated":     list(generated.keys()),
            "error":               gen_error,
        },
        "verification": {
            "passed":     returncode == 0,
            "stdout":     full_stdout,
            "stderr":     full_stderr,
            "returncode": returncode,
        },
        "notebook_ready": notebook_ready,
        "message": (
            "IBM Bob pipeline complete. Notebook ready — "
            "download via /api/download-notebook."
            if notebook_ready
            else "Paper processed — check the terminal for details."
        ),
    })


@app.get("/api/download-notebook")
async def download_notebook():
    """Serve paper2code_pipeline.ipynb as a file attachment."""
    if not NOTEBOOK_PATH.exists():
        raise HTTPException(
            404, "Notebook not generated yet. Upload a paper first."
        )
    return FileResponse(
        path=str(NOTEBOOK_PATH),
        media_type="application/octet-stream",
        filename="paper2code_pipeline.ipynb",
        headers={
            "Content-Disposition": 'attachment; filename="paper2code_pipeline.ipynb"'
        },
    )


@app.get("/api/get-code/{filename}")
async def get_code(filename: str):
    """Return IBM Bob–generated source for model.py, dataset.py, or verify_tensors.py."""
    if filename not in ALLOWED_SRC_FILES:
        raise HTTPException(
            400,
            f"'{filename}' not allowed. Choose from: {sorted(ALLOWED_SRC_FILES)}",
        )
    path = SRC_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found in src/.")
    code_text = path.read_text(encoding="utf-8")
    return JSONResponse({"filename": filename, "code": code_text, "content": code_text})


@app.get("/api/health")
async def health():
    return {
        "status":              "ok",
        "notebook_ready":      NOTEBOOK_PATH.exists(),
        "ibm_bob_configured":  bool(WATSONX_API_KEY and WATSONX_PROJECT),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
