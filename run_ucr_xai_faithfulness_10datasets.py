#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCR XAI faithfulness experiment for the Evidence-Gated SSM paper.

Purpose
-------
This script performs a small, low-cost check before revising the paper toward
an Explainable AI (XAI) contribution. It compares the proposed intrinsic
temporal evidence scores against two cheap post-hoc gradient baselines:

    1. gradient_saliency: |d logit(predicted class) / d x_t|
    2. input_x_gradient: |x_t * d logit(predicted class) / d x_t|

It runs:

    A) Synthetic motif-localization experiment with known ground-truth evidence.
    B) Small UCR pilot on selected datasets, default: ECG5000 and Wafer.

The comparison is explanation-level, not classifier-level:
all methods explain the same trained Proposed_Unnormalized evidence-gated model.

Outputs
-------
Creates timestamped CSV files under Results_Tiny_XAI_Pilot/:

    synthetic_xai_pilot_<timestamp>.csv
    ucr_xai_pilot_detail_<timestamp>.csv
    ucr_xai_pilot_summary_<timestamp>.csv

Example commands
----------------

Synthetic only:
    python run_tiny_xai_risk_pilot.py --mode synthetic --device cpu

UCR pilot only:
    python run_tiny_xai_risk_pilot.py --mode ucr --datasets ECG5000 Wafer --seeds 2025 --device cpu

Both:
    python run_tiny_xai_risk_pilot.py --mode both --device cpu

Notes
-----
This is intentionally small. If the proposed class-logit or margin-logit scores
lose badly to gradient_saliency and input_x_gradient here, the XAI reframing
becomes risky. If they are competitive or better, then it is worth adding
Integrated Gradients and scaling the experiment.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# UCR loader from your existing code base.
# The script expects process_datasets.py to be in the same folder or Python path.
try:
    from process_datasets import load_dataset, get_integer_labels_from_onehot
    PROCESS_DATASETS_AVAILABLE = True
except Exception:
    PROCESS_DATASETS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

INTERNAL_AND_BASELINE_SCORES: Tuple[str, ...] = (
    "random",
    "latent_norm",
    "gate_alpha",
    "evidence_norm",
    "class_logit",
    "margin_logit",
    "gradient_saliency",
    "input_x_gradient",
    "integrated_gradients",
)


TEN_UCR_DATASETS: Tuple[str, ...] = (
    "ECG5000",
    "Wafer",
    "ElectricDevices",
    "FaceAll",
    "PhalangesOutlinesCorrect",
    "CricketX",
    "SwedishLeaf",
    "UWaveGestureLibraryX",
    "Yoga",
    "Earthquakes",
)

DEFAULT_SEEDS: Tuple[int, ...] = (2025, 2026, 2027)

SYNTHETIC_COLUMNS: Tuple[str, ...] = (
    "experiment",
    "seed",
    "method",
    "status",
    "error_message",
    "test_accuracy",
    "test_macro_f1",
    "precision_at_k",
    "recall_at_k",
    "iou",
    "deletion_accuracy",
    "deletion_macro_f1",
    "deletion_accuracy_drop",
    "deletion_macro_f1_drop",
    "insertion_accuracy",
    "insertion_macro_f1",
    "score_time_sec",
    "score_time_sec_per_sample",
    "k",
    "evidence_ratio",
    "n_test_examples",
)

UCR_DETAIL_COLUMNS: Tuple[str, ...] = (
    "experiment",
    "dataset",
    "seed",
    "method",
    "status",
    "error_message",
    "original_accuracy",
    "original_macro_f1",
    "deletion_accuracy",
    "deletion_macro_f1",
    "deletion_accuracy_drop",
    "deletion_macro_f1_drop",
    "insertion_accuracy",
    "insertion_macro_f1",
    "score_time_sec",
    "score_time_sec_per_sample",
    "ratio",
    "k_mean",
    "n_test_examples",
    "best_epoch",
    "train_time_sec",
)

UCR_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "method",
    "n_success",
    "original_accuracy_mean",
    "original_macro_f1_mean",
    "deletion_accuracy_mean",
    "deletion_macro_f1_mean",
    "deletion_accuracy_drop_mean",
    "deletion_macro_f1_drop_mean",
    "insertion_accuracy_mean",
    "insertion_macro_f1_mean",
    "score_time_sec_per_sample_mean",
)


@dataclass
class Config:
    mode: str = "ucr"
    results_dir: Path = Path("Results_UCR_XAI_Faithfulness")
    device: torch.device = torch.device("cpu")

    # Shared training parameters
    hidden_dim: int = 64
    latent_dim: int = 64
    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 0.5
    ig_steps: int = 16

    # Synthetic experiment
    synthetic_seed: int = 2025
    n_samples: int = 3000
    length: int = 100
    motif_len: int = 15
    noise_std: float = 0.50
    motif_amp: float = 2.0
    add_distractor: bool = True
    distractor_amp: float = 2.5
    distractor_len: int = 10
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    synthetic_evidence_ratio: float = 0.10

    # UCR faithfulness experiment
    datasets: List[str] = field(default_factory=lambda: list(TEN_UCR_DATASETS))
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    val_size: float = 0.20
    ucr_ratio: float = 0.10


# =============================================================================
# Utility functions
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_row(csv_path: Path, row: Dict, columns: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    full_row = {col: row.get(col, "") for col in columns}

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def parse_list(value: str, cast=str) -> List:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


# =============================================================================
# Model
# =============================================================================

class EvidenceGatedSSM(nn.Module):
    """
    Proposed unnormalized evidence-gated SSM classifier.

    Input shape:
        x: (batch_size, seq_len)

    Evidence representation:
        z_t = tanh(W_z [h_t, x_t] + b_z)
        alpha_t = sigmoid(w_g^T z_t + b_g)
        e_t = alpha_t z_t
        u = sum_t e_t
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, seq_len), got {tuple(x.shape)}.")

        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list: List[torch.Tensor] = []
        alpha_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            h = h @ self.A.T + x_t @ self.B.T
            z_t = torch.tanh(self.feature_layer(torch.cat([h, x_t], dim=1)))
            alpha_t = torch.sigmoid(self.gate_layer(z_t))

            z_list.append(z_t)
            alpha_list.append(alpha_t)

        z = torch.stack(z_list, dim=1)                  # (B, T, D)
        alpha = torch.stack(alpha_list, dim=1).squeeze(-1)  # (B, T)
        return z, alpha

    def classify_from_evidence(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        evidence = alpha.unsqueeze(-1) * z
        u = evidence.sum(dim=1)
        return self.classifier(u)

    def classify_from_mask(self, z: torch.Tensor, alpha: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        # keep_mask shape: (B, T), values 0 or 1.
        u = ((alpha * keep_mask).unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, alpha = self.encode(x)
        logits = self.classify_from_evidence(z, alpha)
        evidence = alpha.unsqueeze(-1) * z
        return logits, {"z": z, "alpha": alpha, "evidence": evidence}


# =============================================================================
# Data: synthetic
# =============================================================================

def generate_sine_motif(length: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, length)
    return np.sin(t).astype(np.float32)


def generate_synthetic_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.synthetic_seed)

    X = rng.normal(0.0, cfg.noise_std, size=(cfg.n_samples, cfg.length)).astype(np.float32)
    y = rng.integers(0, 2, size=cfg.n_samples).astype(np.int64)
    true_mask = np.zeros((cfg.n_samples, cfg.length), dtype=np.int64)

    motif = generate_sine_motif(cfg.motif_len)

    for i in range(cfg.n_samples):
        if y[i] == 1:
            start = int(rng.integers(5, cfg.length - cfg.motif_len - 5))
            end = start + cfg.motif_len
            X[i, start:end] += cfg.motif_amp * motif
            true_mask[i, start:end] = 1

        if cfg.add_distractor:
            d_start = int(rng.integers(5, cfg.length - cfg.distractor_len - 5))
            d_end = d_start + cfg.distractor_len

            if y[i] == 1:
                attempts = 0
                while true_mask[i, d_start:d_end].sum() > 0 and attempts < 30:
                    d_start = int(rng.integers(5, cfg.length - cfg.distractor_len - 5))
                    d_end = d_start + cfg.distractor_len
                    attempts += 1

            sign = float(rng.choice([-1.0, 1.0]))
            X[i, d_start:d_end] += sign * cfg.distractor_amp

    return X.astype(np.float32), y, true_mask.astype(np.int64)


def split_synthetic(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(cfg.synthetic_seed)
    n = len(y)
    idx = rng.permutation(n)

    n_train = int(cfg.train_ratio * n)
    n_val = int(cfg.val_ratio * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (
        X[train_idx], y[train_idx], mask[train_idx],
        X[val_idx], y[val_idx], mask[val_idx],
        X[test_idx], y[test_idx], mask[test_idx],
    )


# =============================================================================
# DataLoaders
# =============================================================================

def make_loaders_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def make_ucr_loaders(dataset: str, seed: int, cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if not PROCESS_DATASETS_AVAILABLE:
        raise RuntimeError("process_datasets.py could not be imported. Put this script beside process_datasets.py.")

    X_train_full, y_train_oh, X_test, y_test_oh = load_dataset(dataset)
    y_train_full = get_integer_labels_from_onehot(y_train_oh)
    y_test = get_integer_labels_from_onehot(y_test_oh)

    # Ensure CPU tensors before indexing/splitting; batches are moved to cfg.device later.
    X_train_full = X_train_full.detach().cpu().float()
    y_train_full = y_train_full.detach().cpu().long()
    X_test = X_test.detach().cpu().float()
    y_test = y_test.detach().cpu().long()

    n_classes = int(torch.max(torch.cat([y_train_full, y_test])).item() + 1)

    indices = np.arange(len(X_train_full))
    labels = y_train_full.numpy()

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=cfg.val_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=cfg.val_size,
            random_state=seed,
            stratify=None,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train_full[train_idx], y_train_full[train_idx]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(X_train_full[val_idx], y_train_full[val_idx]),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, n_classes


# =============================================================================
# Training and standard evaluation
# =============================================================================

def train_model(
    model: EvidenceGatedSSM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    seed: int,
    label: str,
) -> Tuple[EvidenceGatedSSM, Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_metric = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses: List[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss in {label}, seed={seed}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader, cfg.device)
        val_metric = val_metrics["macro_f1"]

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{label}][seed={seed}] Epoch {epoch:03d} | "
            f"TrainLoss={np.mean(train_losses):.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"ValMacroF1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= cfg.patience:
            print(f"[{label}][seed={seed}] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "train_time_sec": time.time() - start,
    }


def evaluate_loader(model: EvidenceGatedSSM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()

            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# =============================================================================
# Explanation score computation
# =============================================================================

def _internal_scores_from_forward(model: EvidenceGatedSSM, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute logits and internal score tensors.

    Returns:
        logits: (B, C)
        scores: dict mapping method to tensor (B, T)
    """
    logits, aux = model(x)
    z = aux["z"]                         # (B, T, D)
    alpha = aux["alpha"]                 # (B, T)
    evidence = aux["evidence"]           # (B, T, D)

    pred = logits.argmax(dim=1)
    masked_logits = logits.clone()
    masked_logits[torch.arange(len(pred), device=x.device), pred] = -float("inf")
    competitor = masked_logits.argmax(dim=1)

    W = model.classifier.weight          # (C, D)
    W_pred = W[pred]                     # (B, D)
    W_comp = W[competitor]               # (B, D)

    scores = {
        "random": torch.rand_like(alpha),
        "latent_norm": torch.linalg.norm(z, dim=2),
        "gate_alpha": alpha.abs(),
        "evidence_norm": torch.linalg.norm(evidence, dim=2),
        "class_logit": torch.sum(evidence * W_pred.unsqueeze(1), dim=2),
        "margin_logit": torch.sum(evidence * (W_pred - W_comp).unsqueeze(1), dim=2),
    }

    # For ranking, negative contributions can be meaningful, but for a first risk
    # pilot we rank by positive decision support for class_logit/margin_logit.
    # This follows the paper's current top-evidence interpretation.
    scores["class_logit"] = scores["class_logit"]
    scores["margin_logit"] = scores["margin_logit"]

    return logits, scores


def gradient_scores(model: EvidenceGatedSSM, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute gradient saliency and input*gradient scores for a batch.

    Scores are based on the predicted-class logit of the same model.
    """
    model.eval()

    x_grad = x.detach().clone().requires_grad_(True)
    logits, _ = model(x_grad)
    pred = logits.argmax(dim=1)
    selected = logits[torch.arange(x_grad.shape[0], device=x_grad.device), pred].sum()

    model.zero_grad(set_to_none=True)
    if x_grad.grad is not None:
        x_grad.grad.zero_()

    selected.backward()

    grad = x_grad.grad.detach()
    return {
        "gradient_saliency": grad.abs(),
        "input_x_gradient": (x_grad.detach() * grad).abs(),
    }
def integrated_gradients_scores(
    model: EvidenceGatedSSM,
    x: torch.Tensor,
    steps: int = 16,
) -> torch.Tensor:
    """
    Compute Integrated Gradients scores for a batch.

    Baseline:
        zero time series

    Target:
        predicted-class logit of the original input

    Returns:
        Tensor of shape (batch_size, seq_len)
    """
    model.eval()

    x_input = x.detach()
    baseline = torch.zeros_like(x_input)

    with torch.no_grad():
        logits, _ = model(x_input)
        pred = logits.argmax(dim=1)

    total_grad = torch.zeros_like(x_input)

    for step in range(1, steps + 1):
        scale = float(step) / float(steps)
        x_scaled = baseline + scale * (x_input - baseline)
        x_scaled = x_scaled.detach().clone().requires_grad_(True)

        logits_scaled, _ = model(x_scaled)
        selected = logits_scaled[
            torch.arange(x_scaled.shape[0], device=x_scaled.device),
            pred
        ].sum()

        model.zero_grad(set_to_none=True)

        if x_scaled.grad is not None:
            x_scaled.grad.zero_()

        selected.backward()

        total_grad += x_scaled.grad.detach()

    avg_grad = total_grad / float(steps)
    ig = (x_input - baseline) * avg_grad

    return ig.abs()

def compute_scores_for_loader(
    model: EvidenceGatedSSM,
    loader: DataLoader,
    methods: Sequence[str],
    device: torch.device,
    ig_steps: int = 16,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], float]:
    """
    Compute original predictions and explanation score matrices for all examples.

    Returns:
        y_true: shape (N,)
        y_pred: shape (N,)
        scores_by_method: method -> shape (N, T)
        elapsed_sec
    """
    model.eval()

    all_y: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_scores: Dict[str, List[np.ndarray]] = {m: [] for m in methods}

    start = time.time()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)

        # Internal scores need no gradient.
        with torch.no_grad():
            logits, internal = _internal_scores_from_forward(model, X_batch)
            pred = logits.argmax(dim=1)

        all_y.append(y_batch.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        for method in methods:
            if method in internal:
                all_scores[method].append(internal[method].detach().cpu().numpy())

        # Gradient baselines.
        # Gradient-based post-hoc baselines.
        grad_needed = any(m in methods for m in ("gradient_saliency", "input_x_gradient"))
        if grad_needed:
            grads = gradient_scores(model, X_batch)
            for method in ("gradient_saliency", "input_x_gradient"):
                if method in methods:
                    all_scores[method].append(grads[method].detach().cpu().numpy())

        # Integrated Gradients baseline.
        if "integrated_gradients" in methods:
            ig_scores = integrated_gradients_scores(
                model=model,
                x=X_batch,
                steps=ig_steps,
            )
            all_scores["integrated_gradients"].append(
                ig_scores.detach().cpu().numpy()
            )

    elapsed = time.time() - start

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    scores_by_method = {m: np.concatenate(parts, axis=0) for m, parts in all_scores.items() if parts}

    return y_true, y_pred, scores_by_method, elapsed


# =============================================================================
# Deletion/insertion using ranking scores
# =============================================================================

def make_masks_from_scores(
    scores: torch.Tensor,
    k: int,
    operation: str,
    largest: bool = True,
) -> torch.Tensor:
    """
    Create keep masks from temporal scores.

    deletion:
        keep all except top-k selected positions.
    insertion:
        keep only top-k selected positions.

    Args:
        scores: (B, T)
        k: number of selected positions
        operation: "deletion" or "insertion"
        largest: select largest scores if True
    """
    if operation not in {"deletion", "insertion"}:
        raise ValueError(f"Unknown operation: {operation}")

    B, T = scores.shape
    k = max(1, min(int(k), T))

    idx = torch.topk(scores, k=k, dim=1, largest=largest).indices
    selected = torch.zeros(B, T, device=scores.device)
    selected.scatter_(1, idx, 1.0)

    if operation == "deletion":
        return 1.0 - selected
    return selected


def evaluate_masked_by_scores(
    model: EvidenceGatedSSM,
    loader: DataLoader,
    method: str,
    ratio: float,
    device: torch.device,
    ig_steps: int = 16,
) -> Dict[str, float]:
    """
    Compute deletion and insertion performance for one score method.

    For gradient baselines, the temporal indices are selected from input-gradient
    rankings, but deletion/insertion is still performed at the evidence level
    by masking e_t = alpha_t z_t. This keeps the evaluation operation consistent
    across all explanation methods.
    """
    model.eval()

    original_true: List[int] = []
    original_pred: List[int] = []
    deletion_pred: List[int] = []
    insertion_pred: List[int] = []
    k_values: List[int] = []

    start = time.time()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            logits, aux = model(X_batch)
            z = aux["z"]
            alpha = aux["alpha"]
            internal_logits, internal_scores = _internal_scores_from_forward(model, X_batch)
            pred = logits.argmax(dim=1)

        if method in internal_scores:
            scores = internal_scores[method]

        elif method in {"gradient_saliency", "input_x_gradient"}:
            scores = gradient_scores(model, X_batch)[method]

        elif method == "integrated_gradients":
            scores = integrated_gradients_scores(
                model=model,
                x=X_batch,
                steps=ig_steps,
            )

        else:
            raise ValueError(f"Unknown score method: {method}")
        T = scores.shape[1]
        k = max(1, int(round(ratio * T)))
        k_values.append(k)

        del_mask = make_masks_from_scores(scores, k=k, operation="deletion", largest=True)
        ins_mask = make_masks_from_scores(scores, k=k, operation="insertion", largest=True)

        with torch.no_grad():
            del_logits = model.classify_from_mask(z, alpha, del_mask)
            ins_logits = model.classify_from_mask(z, alpha, ins_mask)

        original_true.extend(y_batch.detach().cpu().numpy().tolist())
        original_pred.extend(pred.detach().cpu().numpy().tolist())
        deletion_pred.extend(del_logits.argmax(dim=1).detach().cpu().numpy().tolist())
        insertion_pred.extend(ins_logits.argmax(dim=1).detach().cpu().numpy().tolist())

    elapsed = time.time() - start
    n = len(original_true)

    original_accuracy = accuracy_score(original_true, original_pred)
    original_macro_f1 = f1_score(original_true, original_pred, average="macro", zero_division=0)

    deletion_accuracy = accuracy_score(original_true, deletion_pred)
    deletion_macro_f1 = f1_score(original_true, deletion_pred, average="macro", zero_division=0)

    insertion_accuracy = accuracy_score(original_true, insertion_pred)
    insertion_macro_f1 = f1_score(original_true, insertion_pred, average="macro", zero_division=0)

    return {
        "original_accuracy": original_accuracy,
        "original_macro_f1": original_macro_f1,
        "deletion_accuracy": deletion_accuracy,
        "deletion_macro_f1": deletion_macro_f1,
        "deletion_accuracy_drop": original_accuracy - deletion_accuracy,
        "deletion_macro_f1_drop": original_macro_f1 - deletion_macro_f1,
        "insertion_accuracy": insertion_accuracy,
        "insertion_macro_f1": insertion_macro_f1,
        "score_time_sec": elapsed,
        "score_time_sec_per_sample": elapsed / max(n, 1),
        "k_mean": float(np.mean(k_values)),
        "n_test_examples": n,
    }


# =============================================================================
# Synthetic metrics
# =============================================================================

def localization_metrics(scores: np.ndarray, true_mask: np.ndarray, y_true: np.ndarray, k: int) -> Dict[str, float]:
    """
    Compute Precision@k, Recall@k, and IoU on positive-class test examples only.
    """
    positive_idx = np.where(y_true == 1)[0]

    precisions: List[float] = []
    recalls: List[float] = []
    ious: List[float] = []

    for i in positive_idx:
        score_i = scores[i]
        mask_i = true_mask[i].astype(bool)

        top_idx = np.argsort(score_i)[-k:]
        pred_mask = np.zeros_like(mask_i, dtype=bool)
        pred_mask[top_idx] = True

        inter = np.logical_and(pred_mask, mask_i).sum()
        union = np.logical_or(pred_mask, mask_i).sum()

        precisions.append(inter / max(k, 1))
        recalls.append(inter / max(mask_i.sum(), 1))
        ious.append(inter / max(union, 1))

    return {
        "precision_at_k": safe_mean(precisions),
        "recall_at_k": safe_mean(recalls),
        "iou": safe_mean(ious),
    }


# =============================================================================
# Experiment runners
# =============================================================================

def run_synthetic(cfg: Config, synthetic_csv: Path) -> None:
    print("\n" + "=" * 80)
    print("Running synthetic XAI risk pilot")
    print("=" * 80)

    set_seed(cfg.synthetic_seed)

    X, y, true_mask = generate_synthetic_dataset(cfg)
    split = split_synthetic(X, y, true_mask, cfg)
    X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test = split

    train_loader, val_loader, test_loader = make_loaders_from_arrays(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg.batch_size, cfg.synthetic_seed
    )

    model = EvidenceGatedSSM(
        input_dim=1,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        n_classes=2,
    ).to(cfg.device)

    model, train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        seed=cfg.synthetic_seed,
        label="Synthetic",
    )

    original_metrics = evaluate_loader(model, test_loader, cfg.device)
    print(f"Synthetic test metrics: {original_metrics}")

    # Compute all score matrices once for localization.
    y_true, y_pred, scores_by_method, score_elapsed = compute_scores_for_loader(
        model=model,
        loader=test_loader,
        methods=INTERNAL_AND_BASELINE_SCORES,
        device=cfg.device,
        ig_steps=cfg.ig_steps,
    )

    k = cfg.motif_len

    for method in INTERNAL_AND_BASELINE_SCORES:
        try:
            loc = localization_metrics(scores_by_method[method], mask_test, y_test, k=k)
            masked = evaluate_masked_by_scores(
                model=model,
                loader=test_loader,
                method=method,
                ratio=cfg.synthetic_evidence_ratio,
                device=cfg.device,
                ig_steps=cfg.ig_steps,
            )

            row = {
                "experiment": "synthetic",
                "seed": cfg.synthetic_seed,
                "method": method,
                "status": "success",
                "error_message": "",
                "test_accuracy": original_metrics["accuracy"],
                "test_macro_f1": original_metrics["macro_f1"],
                "precision_at_k": loc["precision_at_k"],
                "recall_at_k": loc["recall_at_k"],
                "iou": loc["iou"],
                "deletion_accuracy": masked["deletion_accuracy"],
                "deletion_macro_f1": masked["deletion_macro_f1"],
                "deletion_accuracy_drop": masked["deletion_accuracy_drop"],
                "deletion_macro_f1_drop": masked["deletion_macro_f1_drop"],
                "insertion_accuracy": masked["insertion_accuracy"],
                "insertion_macro_f1": masked["insertion_macro_f1"],
                "score_time_sec": masked["score_time_sec"],
                "score_time_sec_per_sample": masked["score_time_sec_per_sample"],
                "k": k,
                "evidence_ratio": cfg.synthetic_evidence_ratio,
                "n_test_examples": len(y_test),
            }
        except Exception as exc:
            row = {
                "experiment": "synthetic",
                "seed": cfg.synthetic_seed,
                "method": method,
                "status": "failed",
                "error_message": str(exc),
            }

        append_row(synthetic_csv, row, SYNTHETIC_COLUMNS)
        print("Synthetic row:", row)


def run_ucr(cfg: Config, detail_csv: Path, summary_csv: Path) -> None:
    print("\n" + "=" * 80)
    print("Running UCR XAI risk pilot")
    print("=" * 80)

    if not PROCESS_DATASETS_AVAILABLE:
        raise RuntimeError("process_datasets.py could not be imported. UCR mode cannot run.")

    rows_for_summary: List[Dict] = []

    for dataset in cfg.datasets:
        for seed in cfg.seeds:
            print("\n" + "-" * 80)
            print(f"Dataset={dataset}, seed={seed}")
            print("-" * 80)

            set_seed(seed)

            try:
                train_loader, val_loader, test_loader, n_classes = make_ucr_loaders(dataset, seed, cfg)

                model = EvidenceGatedSSM(
                    input_dim=1,
                    hidden_dim=cfg.hidden_dim,
                    latent_dim=cfg.latent_dim,
                    n_classes=n_classes,
                ).to(cfg.device)

                model, train_info = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=cfg,
                    seed=seed,
                    label=dataset,
                )

                for method in INTERNAL_AND_BASELINE_SCORES:
                    try:
                        masked = evaluate_masked_by_scores(
                            model=model,
                            loader=test_loader,
                            method=method,
                            ratio=cfg.ucr_ratio,
                            device=cfg.device,
                            ig_steps=cfg.ig_steps,
                        )

                        row = {
                            "experiment": "ucr",
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "status": "success",
                            "error_message": "",
                            **masked,
                            "ratio": cfg.ucr_ratio,
                            "best_epoch": train_info["best_epoch"],
                            "train_time_sec": train_info["train_time_sec"],
                        }
                    except Exception as exc:
                        row = {
                            "experiment": "ucr",
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "status": "failed",
                            "error_message": str(exc),
                            "ratio": cfg.ucr_ratio,
                            "best_epoch": train_info.get("best_epoch", ""),
                            "train_time_sec": train_info.get("train_time_sec", ""),
                        }

                    append_row(detail_csv, row, UCR_DETAIL_COLUMNS)
                    rows_for_summary.append(row)
                    print("UCR row:", row)

            except Exception as exc:
                for method in INTERNAL_AND_BASELINE_SCORES:
                    row = {
                        "experiment": "ucr",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "status": "failed",
                        "error_message": str(exc),
                        "ratio": cfg.ucr_ratio,
                    }
                    append_row(detail_csv, row, UCR_DETAIL_COLUMNS)
                    rows_for_summary.append(row)
                    print("UCR failed row:", row)

    write_ucr_summary(rows_for_summary, summary_csv)


def write_ucr_summary(rows: List[Dict], summary_csv: Path) -> None:
    groups: Dict[Tuple[str, str], List[Dict]] = {}

    for row in rows:
        if row.get("status") != "success":
            continue
        key = (str(row.get("dataset")), str(row.get("method")))
        groups.setdefault(key, []).append(row)

    for (dataset, method), group in sorted(groups.items()):
        summary = {
            "dataset": dataset,
            "method": method,
            "n_success": len(group),
            "original_accuracy_mean": safe_mean(float(r["original_accuracy"]) for r in group),
            "original_macro_f1_mean": safe_mean(float(r["original_macro_f1"]) for r in group),
            "deletion_accuracy_mean": safe_mean(float(r["deletion_accuracy"]) for r in group),
            "deletion_macro_f1_mean": safe_mean(float(r["deletion_macro_f1"]) for r in group),
            "deletion_accuracy_drop_mean": safe_mean(float(r["deletion_accuracy_drop"]) for r in group),
            "deletion_macro_f1_drop_mean": safe_mean(float(r["deletion_macro_f1_drop"]) for r in group),
            "insertion_accuracy_mean": safe_mean(float(r["insertion_accuracy"]) for r in group),
            "insertion_macro_f1_mean": safe_mean(float(r["insertion_macro_f1"]) for r in group),
            "score_time_sec_per_sample_mean": safe_mean(float(r["score_time_sec_per_sample"]) for r in group),
        }
        append_row(summary_csv, summary, UCR_SUMMARY_COLUMNS)


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a UCR XAI faithfulness experiment: intrinsic evidence scores vs gradient saliency and input×gradient."
    )

    parser.add_argument("--mode", choices=["synthetic", "ucr", "both"], default="ucr")
    parser.add_argument("--results-dir", type=str, default="Results_UCR_XAI_Faithfulness")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=0.5)

    # Synthetic
    parser.add_argument("--synthetic-seed", type=int, default=2025)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--motif-len", type=int, default=15)
    parser.add_argument("--noise-std", type=float, default=0.50)
    parser.add_argument("--motif-amp", type=float, default=2.0)
    parser.add_argument("--no-distractor", action="store_true")
    parser.add_argument("--distractor-amp", type=float, default=2.5)
    parser.add_argument("--distractor-len", type=int, default=10)
    parser.add_argument("--synthetic-evidence-ratio", type=float, default=0.10)

    # UCR
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(TEN_UCR_DATASETS),
        help="Comma-separated UCR dataset names. Default: the ten selected UCR datasets."
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds. Default: 2025,2026,2027."
    )
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--ucr-ratio", type=float, default=0.10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        mode=args.mode,
        results_dir=Path(args.results_dir),
        device=resolve_device(args.device),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        synthetic_seed=args.synthetic_seed,
        n_samples=args.n_samples,
        length=args.length,
        motif_len=args.motif_len,
        noise_std=args.noise_std,
        motif_amp=args.motif_amp,
        add_distractor=not args.no_distractor,
        distractor_amp=args.distractor_amp,
        distractor_len=args.distractor_len,
        synthetic_evidence_ratio=args.synthetic_evidence_ratio,
        datasets=parse_list(args.datasets, str),
        seeds=parse_list(args.seeds, int),
        val_size=args.val_size,
        ucr_ratio=args.ucr_ratio,
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    synthetic_csv = cfg.results_dir / f"synthetic_xai_pilot_{timestamp}.csv"
    ucr_detail_csv = cfg.results_dir / f"ucr_xai_pilot_detail_{timestamp}.csv"
    ucr_summary_csv = cfg.results_dir / f"ucr_xai_pilot_summary_{timestamp}.csv"

    print("UCR XAI faithfulness experiment")
    print("Mode:", cfg.mode)
    print("Device:", cfg.device)
    print("Results directory:", cfg.results_dir)
    print("Methods:", INTERNAL_AND_BASELINE_SCORES)

    if cfg.mode in {"synthetic", "both"}:
        run_synthetic(cfg, synthetic_csv)
        print("Synthetic CSV:", synthetic_csv)

    if cfg.mode in {"ucr", "both"}:
        run_ucr(cfg, ucr_detail_csv, ucr_summary_csv)
        print("UCR detail CSV:", ucr_detail_csv)
        print("UCR summary CSV:", ucr_summary_csv)

    print("\nFinished UCR XAI faithfulness experiment.")


if __name__ == "__main__":
    main()
