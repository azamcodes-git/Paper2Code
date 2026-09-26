"""
Dataset and DataLoader for hyperspectral beverage-stain classification.

Paper: "A hyperspectral imaging framework integrating band selection and deep
       learning for beverage stain classification in forensic analysis"
       Scientific Reports (2026) 16:22241

Dataset facts (from paper):
  - Raw hyperspectral cube:  512 × 512 × 204  (spatial X, spatial Y, spectral bands)
  - After quality filtering:  117,339 spectral samples × 204 bands
  - After ANOVA band selection: 117,339 samples × 162 bands  ← model input
  - Preprocessing: StandardScaler (zero mean, unit variance) fitted on training data only
  - Split: 80 % train / 20 % test (stratified by class)
           + 10 % of training set as validation ("validation_split=0.1")
           → effective proportions: ~72 % train, ~8 % val, ~20 % test
  - Batch size: 64
  - Labels: 0=Papaya, 1=Coffee, 2=Pomegranate, 3=Orange, 4=Tea,
            5=Wine, 6=Whisky, 7=Rum, 8=Brandy

Expected data file layout (CSV exported from Specim IQ):
  - 54 CSV files merged into one master CSV
  - Columns: band_0, band_1, ..., band_203, label
  - After ANOVA selection: 162 band columns + 1 label column
  - Rows are individual pixel spectral samples
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------
NUM_BANDS = 162       # ANOVA-selected spectral bands (model input width)
NUM_CLASSES = 9       # beverage stain categories
BATCH_SIZE = 64       # Table 3
VAL_FRACTION = 0.10   # 10 % of training split held for validation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BeverageStainDataset(Dataset):
    """
    Pixel-wise spectral dataset for hyperspectral beverage stain classification.

    Each sample is a 1-D reflectance vector of shape (162,) and an integer label
    in [0, 8].  The class operates on pre-loaded numpy arrays so it works whether
    data comes from a CSV file or is already in memory.

    Args:
        features : float32 array of shape (N, 162) — StandardScaler-normalised.
        labels   : int64  array of shape (N,)      — class indices in [0, 8].
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert features.ndim == 2 and features.shape[1] == NUM_BANDS, (
            f"Expected features shape (N, {NUM_BANDS}), got {features.shape}"
        )
        assert labels.ndim == 1 and len(features) == len(labels), (
            "features and labels must have the same number of rows"
        )
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels   = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the merged master CSV produced by the paper's preprocessing pipeline.

    Expected format: 162 numeric band columns followed by a 'label' column.
    The label column contains integer class indices 0–8.

    Returns:
        features : float32 ndarray of shape (N, 162)
        labels   : int64   ndarray of shape (N,)
    """
    import pandas as pd  # optional dependency; only needed for CSV loading

    df = pd.read_csv(path)
    label_col = "label"
    if label_col not in df.columns:
        raise ValueError(
            f"CSV must contain a '{label_col}' column. Found: {list(df.columns)}"
        )
    labels   = df[label_col].to_numpy(dtype=np.int64)
    features = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)

    if features.shape[1] != NUM_BANDS:
        raise ValueError(
            f"Expected {NUM_BANDS} feature columns after dropping 'label', "
            f"got {features.shape[1]}. Run ANOVA band selection first."
        )
    return features, labels


# ---------------------------------------------------------------------------
# Preprocessing: StandardScaler fitted on training data only
# ---------------------------------------------------------------------------
class SpectralScaler:
    """
    Zero-mean / unit-variance normalisation over spectral features.

    Fitted exclusively on training samples to prevent information leakage
    (paper: "Input normalization: StandardScaler (zero mean, unit variance)").
    """

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SpectralScaler":
        """Compute mean and std from training data X (N, 162)."""
        self._mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        self._std  = X.std(axis=0,  keepdims=True).astype(np.float32)
        self._std  = np.where(self._std == 0, 1.0, self._std)  # avoid /0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None, "Call fit() before transform()"
        return ((X - self._mean) / self._std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, mean=self._mean, std=self._std)

    def load(self, path: str) -> "SpectralScaler":
        data = np.load(path)
        self._mean, self._std = data["mean"], data["std"]
        return self


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def make_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from pre-normalised numpy arrays.

    Splits are stratified at the caller side; this function applies a random
    val_fraction split on whatever training data is passed in.

    Returns:
        train_loader, val_loader
    """
    dataset   = BeverageStainDataset(features, labels)
    n_val     = int(len(dataset) * val_fraction)
    n_train   = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  drop_last=False
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader


def make_test_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build a test DataLoader (no shuffling, no dropping)."""
    dataset = BeverageStainDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# ---------------------------------------------------------------------------
# End-to-end helper: load CSV → scale → split → DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders_from_csv(
    csv_path: str,
    test_size: float = 0.20,
    batch_size: int = BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SpectralScaler]:
    """
    One-call helper: CSV → stratified train/val/test DataLoaders + fitted scaler.

    Replicates the paper's 80:20 stratified split + 10 % validation hold-out.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    from sklearn.model_selection import train_test_split  # optional dep

    features, labels = load_csv(csv_path)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Fit scaler on training data only
    scaler  = SpectralScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_loader, val_loader = make_dataloaders(
        X_train, y_train, batch_size=batch_size, val_fraction=val_fraction, seed=seed
    )
    test_loader = make_test_loader(X_test, y_test, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler
