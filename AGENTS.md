# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Context

IBM Watsonx Hackathon template. Stack is Python ML (`src/train.py`, `src/model.py`, `src/dataset.py`) — all three files are currently empty stubs. Research papers go in `papers/`.

## Build / Run

No build system or package manager config exists yet. Run training directly:

```bash
python src/train.py
```

No test framework, lint config, or CI is configured. Add these as the project grows.

## Critical Security Constraints

These are enforced by `.gitignore` and `.bobignore` — **do not work around them**:

- **Never hardcode credentials.** All secrets must come from environment variables loaded via `python-dotenv`: `os.getenv('IBM_CLOUD_API_KEY')`.
- **Never read or suggest reading `.env`** — it is gitignored and bobignored. Tell users to add credentials themselves.
- **Never include credential values in prompts or code snippets** — Bob session history may log them. Use placeholder names only (e.g. `<YOUR_API_KEY>`).
- Files whose names match `*secret*`, `*credential*`, `*password*`, `*token*`, `config.json`, `config.yaml` are gitignored — do not create files with these names.
- `bob_sessions/` exports **must** be committed for project submission (live session files are excluded, exported reports are not).

## Code Style

No linter config present. Follow standard Python conventions (PEP 8). Load env vars at module top level using `python-dotenv`.
