# Paper2Code — IBM Bob 2.0 Hackathon

Upload a Computer Vision research paper (PDF) → IBM Bob verifies the neural-network tensors → view the generated PyTorch source (`model.py`, `dataset.py`) in the browser.

---

## Project Structure

```
Paper2Code/
├── papers/                  # Uploaded PDF files land here (git-ignored binaries)
│   └── paper.pdf            # Sample paper (hyperspectral beverage stain, Sci. Rep. 2026)
│
├── src/
│   ├── main.py              # ★ FastAPI backend — start here
│   ├── index.html           # Minimalist frontend served at http://localhost:8000
│   ├── model.py             # PyTorch model definitions (MLP, 1D-CNN, LSTM, CNN-LSTM)
│   ├── dataset.py           # Dataset, DataLoader, SpectralScaler
│   ├── verify_tensors.py    # Automated tensor-shape verification sandbox
│   └── train.py             # (stub) training loop — not yet implemented
│
├── requirements.txt         # Python dependencies
├── AGENTS.md                # Rules for IBM Bob AI assistant
├── CONTRIBUTING.md          # API contract + frontend dev guide  ← read this next
└── .env                     # NOT committed — create from .env.example
```

---

## Quick Start

### 1 — Clone & install

```bash
git clone <repo-url>
cd Paper2Code
pip install -r requirements.txt
```

### 2 — Start the server

```bash
py src/main.py          # Windows
python src/main.py      # macOS / Linux
```

Server starts at **http://localhost:8000**

> ⚠️ Always open the UI at `http://localhost:8000`.  
> Do **not** open `src/index.html` directly in a browser or via VS Code Live Server —  
> the API calls will fail because they target port 8000.

### 3 — Use the app

1. Go to **http://localhost:8000**
2. Choose a PDF research paper and click **Upload & Process**
3. The verification log shows model tensor checks (all 5 should show `[PASS]`)
4. Switch tabs to view `model.py`, `dataset.py`, or `verify_tensors.py`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serves `index.html` |
| `GET`  | `/api/health` | Returns `{"status":"ok"}` |
| `POST` | `/api/upload-paper` | Upload a PDF; triggers verification; returns JSON logs |
| `GET`  | `/api/get-code/{filename}` | Returns source of `model.py`, `dataset.py`, or `verify_tensors.py` |

Full request/response shapes are documented in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| ML framework | PyTorch |
| PDF handling | pypdf |
| Data processing | NumPy |
| Env / secrets | python-dotenv |
| Frontend (current) | Vanilla HTML/JS (teammate replacing with full UI) |

---

## Environment Variables

Create a `.env` file in the repo root (it is git-ignored):

```
PORT=8000           # optional — default is 8000
```

IBM Watsonx / Bob credentials go here when integrating the AI pipeline. Load them with:

```python
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("IBM_CLOUD_API_KEY")
```

---

## Running Verification Standalone

```bash
py src/verify_tensors.py
```

Expected output — all 5 checks passing:

```
  [PASS] MLP         input=(8, 162)  output=(8, 9)
  [PASS] 1D-CNN      input=(8, 162)  output=(8, 9)
  [PASS] LSTM        input=(8, 162)  output=(8, 9)
  [PASS] CNN-LSTM    input=(8, 162)  output=(8, 9)
  [PASS] BeverageStainDataset  item=(feat(162,), label())  len=16
  All 5 checks passed. OK
```

---

## Security

- Never commit `.env` or any file containing credentials.
- See [SECURITY.MD](SECURITY.MD) for full guidelines.
- See [AGENTS.md](AGENTS.md) for IBM Bob AI assistant rules.
