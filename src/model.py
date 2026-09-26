"""
Neural network architectures from:
  "A hyperspectral imaging framework integrating band selection and deep learning
   for beverage stain classification in forensic analysis"
  Scientific Reports (2026) 16:22241

Input: (N, 162) float32 spectral vectors (162 ANOVA-selected bands, StandardScaler-normalised)
Output: (N, 9)  logits over 9 beverage classes

Hyperparameters (Table 3):
  - Optimizer : Adam(lr=0.001, betas=(0.9, 0.999))
  - LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5)
  - Dropout rate : 0.3
  - Batch size   : 64
  - Max epochs   : 100
  - Early stopping patience: 10 (monitor val_loss)
  - Loss function: CrossEntropyLoss (categorical cross-entropy)
"""

import torch
import torch.nn as nn

NUM_BANDS = 162   # ANOVA-selected spectral features
NUM_CLASSES = 9   # beverage stain categories
DROPOUT = 0.3


# ---------------------------------------------------------------------------
# 1. MLP  (best model, 95.58% accuracy; ~230 k trainable params)
#    Architecture (Table 3 + Eq. 6-8):
#      Linear(162→512) → ReLU → Dropout(0.3)
#      Linear(512→256) → ReLU → Dropout(0.3)
#      Linear(256→128) → ReLU → Dropout(0.3)
#      Linear(128→9)   → (loss: CrossEntropyLoss, pred: Softmax)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Multi-Layer Perceptron for 1-D spectral vectors."""

    def __init__(self, num_bands: int = NUM_BANDS, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_bands, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes),  # raw logits; Softmax applied by loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 162)
        return self.network(x)


# ---------------------------------------------------------------------------
# 2. 1D-CNN  (~120 k trainable params)
#    Architecture (Table 3 + Eq. 9-10):
#      Input unsqueezed to (N, 1, 162)  — channel dimension for Conv1d
#      Conv1d(1→64,  k=3, pad=1) → ReLU → MaxPool1d(2)   → (N, 64,  81)
#      Conv1d(64→128, k=3, pad=1) → ReLU → MaxPool1d(2)  → (N, 128, 40)
#      Flatten → Linear(5120→128) → ReLU → Dropout(0.3)
#      Linear(128→9)
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    """1-D Convolutional Neural Network over the spectral dimension."""

    def __init__(self, num_bands: int = NUM_BANDS, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        # After two MaxPool1d(2): floor(floor(num_bands/2)/2)
        reduced = (num_bands // 2) // 2          # = 40 for num_bands=162
        flat_dim = 128 * reduced                 # = 5120

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 162)  →  unsqueeze to (N, 1, 162) for Conv1d
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. LSTM  (~150 k trainable params)
#    Architecture (Table 3 + Eq. 11-16):
#      Input reshaped to (N, 162, 1)  — sequence of 162 scalar band values
#      LSTM(input_size=1, hidden_size=128, batch_first=True)
#      Take final hidden state h_T: (N, 128)
#      Dropout(0.3) → Linear(128→9)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM treating the spectral axis as a time sequence."""

    def __init__(self, num_bands: int = NUM_BANDS, num_classes: int = NUM_CLASSES,
                 hidden_size: int = 128, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 162)  →  (N, 162, 1) as sequence of 162 time-steps, each of dim 1
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)   # h_n: (1, N, 128)
        return self.head(h_n.squeeze(0))


# ---------------------------------------------------------------------------
# 4. CNN-LSTM hybrid  (~95 k trainable params)
#    Architecture (Table 3 + Eq. 17-19):
#      Input unsqueezed to (N, 1, 162)
#      Conv1d(1→64, k=3, pad=1) → ReLU → MaxPool1d(2)  → (N, 64, 81)
#      Permute to (N, 81, 64)   — LSTM expects (batch, seq, features)
#      LSTM(input_size=64, hidden_size=64, batch_first=True)
#      Take final hidden state: (N, 64)
#      Dropout(0.3) → Linear(64→9)
# ---------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM: CNN extracts local spectral features, LSTM models sequence."""

    def __init__(self, num_bands: int = NUM_BANDS, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),           # (N, 64, num_bands//2)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 162)
        x = x.unsqueeze(1)                         # (N, 1, 162)
        x = self.cnn(x)                            # (N, 64, 81)
        x = x.permute(0, 2, 1)                    # (N, 81, 64) — seq, features
        _, (h_n, _) = self.lstm(x)                # h_n: (1, N, 64)
        return self.head(h_n.squeeze(0))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "mlp": MLP,
    "1d-cnn": CNN1D,
    "lstm": LSTMModel,
    "cnn-lstm": CNNLSTM,
}

BEVERAGE_CLASSES = [
    "Papaya",       # 0
    "Coffee",       # 1
    "Pomegranate",  # 2
    "Orange",       # 3
    "Tea",          # 4
    "Wine",         # 5
    "Whisky",       # 6
    "Rum",          # 7
    "Brandy",       # 8
]


def build_model(name: str = "mlp", **kwargs) -> nn.Module:
    """Return an initialised model by name: 'mlp', '1d-cnn', 'lstm', 'cnn-lstm'."""
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


def build_optimizer(model: nn.Module, lr: float = 0.001) -> torch.optim.Adam:
    """Adam(beta1=0.9, beta2=0.999) as specified in Table 3."""
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Adam) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """ReduceLROnPlateau(factor=0.5, patience=5) as specified in Table 3."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
