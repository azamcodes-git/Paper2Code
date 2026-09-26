"""
Tensor dimensionality verification script.

Instantiates all four model architectures from the paper, passes a synthetic
batch of random spectral vectors through each one, and confirms output shapes
are correct.  Run with:

    py src/verify_tensors.py

Expected output (no errors means all shapes are correct):

    ── Tensor verification ──────────────────────────────────
    Input tensor : (8, 162)  float32
    ─────────────────────────────────────────────────────────
    [PASS] MLP       input=(8, 162)  output=(8, 9)   params=~230,505
    [PASS] 1D-CNN    input=(8, 162)  output=(8, 9)   params=~121,929
    [PASS] LSTM      input=(8, 162)  output=(8, 9)   params=~141,833
    [PASS] CNN-LSTM  input=(8, 162)  output=(8, 9)   params=~98,633
    ─────────────────────────────────────────────────────────
    All 4 models passed dimensionality checks.
"""

import sys
import os

# Allow running from either repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model import MLP, CNN1D, LSTMModel, CNNLSTM, NUM_BANDS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Synthetic input matching exact paper dimensions
# ---------------------------------------------------------------------------
BATCH_SIZE = 8        # small batch for quick checks
NUM_BANDS  = 162      # ANOVA-selected spectral bands  (paper Table 3 / Sec. 3.7)
NUM_CLASSES = 9       # beverage stain categories

# Simulate StandardScaler-normalised spectral vectors  (zero mean, unit variance)
torch.manual_seed(0)
x_synthetic = torch.randn(BATCH_SIZE, NUM_BANDS, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
EXPECTED_OUTPUT_SHAPE = (BATCH_SIZE, NUM_CLASSES)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify(name: str, model: torch.nn.Module, x: torch.Tensor) -> bool:
    model.eval()
    with torch.no_grad():
        try:
            y = model(x)
        except Exception as exc:
            print(f"  [FAIL] {name:<10}  ERROR during forward pass: {exc}")
            return False

    if y.shape != torch.Size(EXPECTED_OUTPUT_SHAPE):
        print(
            f"  [FAIL] {name:<10}  output shape {tuple(y.shape)} "
            f"!= expected {EXPECTED_OUTPUT_SHAPE}"
        )
        return False

    n = count_params(model)
    print(
        f"  [PASS] {name:<10}  input={tuple(x.shape)}  "
        f"output={tuple(y.shape)}   params=~{n:,}"
    )
    return True


# ---------------------------------------------------------------------------
# Dataset shape sanity-check (no real data needed)
# ---------------------------------------------------------------------------
def verify_dataset() -> bool:
    """Create a tiny in-memory BeverageStainDataset and check __getitem__ shapes."""
    import numpy as np
    from src.dataset import BeverageStainDataset, NUM_BANDS as DS_BANDS

    n = 16
    fake_features = np.random.randn(n, DS_BANDS).astype(np.float32)
    fake_labels   = np.random.randint(0, 9, size=(n,), dtype=np.int64)

    try:
        ds = BeverageStainDataset(fake_features, fake_labels)
        feat, label = ds[0]
        assert feat.shape == (DS_BANDS,),  f"Feature shape {feat.shape} != ({DS_BANDS},)"
        assert label.shape == (),          f"Label shape {label.shape} != scalar"
        assert feat.dtype  == torch.float32
        assert label.dtype == torch.int64
        print(
            f"  [PASS] BeverageStainDataset  "
            f"item=(feat{tuple(feat.shape)}, label{tuple(label.shape)})  "
            f"len={len(ds)}"
        )
        return True
    except Exception as exc:
        print(f"  [FAIL] BeverageStainDataset  {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sep = "-" * 57
    print(f"\n{sep}")
    print("  Tensor verification - paper: Sci. Rep. 16:22241 (2026)")
    print(sep)
    print(f"  Input tensor : {tuple(x_synthetic.shape)}  {x_synthetic.dtype}")
    print(sep)

    results = []

    # All four architectures from Table 3
    checks = [
        ("MLP",      MLP()),
        ("1D-CNN",   CNN1D()),
        ("LSTM",     LSTMModel()),
        ("CNN-LSTM", CNNLSTM()),
    ]

    for name, model in checks:
        results.append(verify(name, model, x_synthetic))

    print(sep)

    # Dataset sanity
    print("  Dataset item shapes:")
    results.append(verify_dataset())
    print(sep)

    n_pass = sum(results)
    n_total = len(results)

    if n_pass == n_total:
        print(f"  All {n_total} checks passed. OK")
    else:
        print(f"  {n_pass}/{n_total} checks passed - see FAIL lines above.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
