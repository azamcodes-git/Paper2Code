# Contributing — Frontend & API Guide

This document is for the **frontend / full-stack teammate**. It covers every API endpoint, its exact request/response shape, and notes on how to replace the placeholder `index.html` with a proper frontend.

---

## 1. Running the Backend Locally

```bash
pip install -r requirements.txt
py src/main.py          # Windows
python src/main.py      # macOS / Linux
```

Server: **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs** (FastAPI auto-generated Swagger UI)

---

## 2. API Contract

All responses are `application/json` unless noted otherwise.

---

### `GET /`

Returns the HTML frontend page (`src/index.html`).  
Replace this file with your own — or point this route to a different file / SPA build output.

```
Response: text/html  200 OK
```

---

### `GET /api/health`

Liveness probe — call this to confirm the server is up.

```json
// 200 OK
{ "status": "ok" }
```

---

### `POST /api/upload-paper`

Upload a research paper PDF and trigger the automated verification sandbox.

**Request** — `multipart/form-data`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | file | ✅ | Must be a `.pdf` file |

**Example (fetch)**
```js
const fd = new FormData();
fd.append('file', fileInputElement.files[0]);
const res = await fetch('http://localhost:8000/api/upload-paper', {
  method: 'POST',
  body: fd,
});
const data = await res.json();
```

**Success response** — `200 OK`
```json
{
  "status": "ok",
  "filename": "paper.pdf",
  "saved_to": "papers/paper.pdf",
  "verification": {
    "passed": true,
    "stdout": "  [PASS] MLP  ...\n  All 5 checks passed. OK\n",
    "stderr": "",
    "returncode": 0
  },
  "message": "Paper saved and verification passed. Retrieve code via /api/get-code/{filename}."
}
```

**Error response** — `200 OK` with `status: "error"` (verification failed)
```json
{
  "status": "error",
  "filename": "paper.pdf",
  "saved_to": "papers/paper.pdf",
  "verification": {
    "passed": false,
    "stdout": "",
    "stderr": "...",
    "returncode": 1
  },
  "message": "Paper saved but verification reported errors — see logs."
}
```

**Validation error** — `400 Bad Request` (non-PDF uploaded)
```json
{ "detail": "Only PDF files are accepted." }
```

---

### `GET /api/get-code/{filename}`

Retrieve the source code of a generated Python file.

**Allowed values for `{filename}`:**

| filename | Description |
|----------|-------------|
| `model.py` | PyTorch model architectures (MLP, 1D-CNN, LSTM, CNN-LSTM) |
| `dataset.py` | Dataset class, DataLoader factory, SpectralScaler |
| `verify_tensors.py` | Verification sandbox script |

**Example (fetch)**
```js
const res  = await fetch('http://localhost:8000/api/get-code/model.py');
const data = await res.json();
console.log(data.content); // full source code string
```

**Success response** — `200 OK`
```json
{
  "filename": "model.py",
  "content": "\"\"\"\\nNeural network architectures from: ...\\n\"\"\"\\nimport torch\\n..."
}
```

**Error responses**
```json
// 400 — filename not in allowed list
{ "detail": "'train.py' is not an allowed file. Choose from: ['dataset.py', 'model.py', 'verify_tensors.py']" }

// 404 — file missing from src/
{ "detail": "model.py not found in src/." }
```

---

## 3. Replacing the Frontend

The current `src/index.html` is a minimal placeholder — your job is to replace it with a proper UI.

**Option A — Replace `src/index.html` in place**  
The `GET /` route in `src/main.py` serves `src/index.html` directly via `FileResponse`. Just overwrite that file with your new HTML/CSS/JS (single-page file).

**Option B — Point to a separate build output (React / Vue / etc.)**  
In [`src/main.py`](src/main.py), find the `index()` route and change the path:
```python
# src/main.py  ~line 49
html_path = SRC_DIR / "index.html"           # ← change this
# e.g. for a Vite build output:
# html_path = BASE_DIR / "frontend" / "dist" / "index.html"
```
Then also mount the static assets directory:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "frontend" / "dist" / "assets")), name="assets")
```

**Option C — Separate dev server with CORS**  
Add CORS middleware to `src/main.py` so your Vite/Next dev server can call the FastAPI backend:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # your dev server port
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 4. Key Integration Points for the Frontend

| What to display | Where to get it |
|----------------|----------------|
| Verification log (pass/fail + stdout) | `POST /api/upload-paper` → `data.verification.stdout` |
| Overall status | `data.status` — `"ok"` or `"error"` |
| `model.py` source | `GET /api/get-code/model.py` → `data.content` |
| `dataset.py` source | `GET /api/get-code/dataset.py` → `data.content` |
| Is server alive? | `GET /api/health` → `{"status":"ok"}` |

---

## 5. File Ownership

| File | Owner | Notes |
|------|-------|-------|
| `src/main.py` | Backend | Do not edit routes without coordinating |
| `src/index.html` | Frontend | Replace freely |
| `src/model.py` | Backend/AI | Auto-generated — do not hand-edit |
| `src/dataset.py` | Backend/AI | Auto-generated — do not hand-edit |
| `src/verify_tensors.py` | Backend | Verification sandbox — do not edit |
| `src/train.py` | Backend | Stub — not yet implemented |
| `requirements.txt` | Backend | Add frontend build tools separately |
| `papers/` | Shared | Auto-populated on upload — gitignored |
