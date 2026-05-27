from __future__ import annotations

from datetime import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from process_datasets import (
    load_dataset,
    get_integer_labels_from_onehot,
    device,
)

# ============================================================
# Configuration
# ============================================================

# Laptop:
RESULTS_DIR = Path("Results")
# Colab:
# RESULTS_DIR = Path("/content/drive/MyDrive/Results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    # Recommended qualitative examples.
    # ECG5000 and ElectricDevices usually show clearer deletion behavior.
    dataset_names: List[str] = field(default_factory=lambda: [
        "ECG5000",
        "ElectricDevices",
        "FaceAll",
    ])

    seed: int = 2025

    model_name: str = "Proposed_Unnormalized_Base"

    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 50
    patience: int = 10
    hidden_dim: int = 64
    latent_dim: int = 64
    clip_grad_max_norm: float = 0.5

    use_validation_split: bool = True
    val_size: float = 0.2

    # Qualitative evidence profile settings.
    top_fraction: float = 0.10
    delete_fraction: float = 0.10

    # Number of qualitative examples per dataset.
    n_examples_per_dataset: int = 2

    # Candidate selection.
    # "largest_confidence_drop": choose correctly classified examples whose confidence
    # drops most after deleting the top classifier-aware evidence terms.
    # "highest_confidence": choose correctly classified examples with highest clean confidence.
    selection_rule: str = "largest_confidence_drop"

    # Which score is used to select/highlight/delete top evidence terms.
    # Recommended: "class_logit" because it was strongest in the UCR faithfulness table.
    # Alternatives: "margin_logit", "evidence_norm", "gate_alpha".
    selection_score: str = "class_logit"

    output_dir: str = ""


# ============================================================
# Utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"classifier_aware_qualitative_profiles_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    seed: int,
    use_validation_split: bool,
    val_size: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if use_validation_split:
        indices = np.arange(len(X_train))
        y_np = y_train.detach().cpu().numpy()

        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=y_np,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=None,
            )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
    else:
        X_tr = X_train
        y_tr = y_train
        X_val = X_test
        y_val = y_test

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Model
# ============================================================

class ProposedUnnormalizedSSM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:     shape (batch_size, seq_len, latent_dim)
            alpha: shape (batch_size, seq_len)
        """
        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list = []
        alpha_list = []

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            h = h @ self.A.T + x_t @ self.B.T
            z_t = torch.tanh(self.feature_layer(torch.cat([h, x_t], dim=1)))
            alpha_t = torch.sigmoid(self.gate_layer(z_t))

            z_list.append(z_t)
            alpha_list.append(alpha_t)

        z = torch.stack(z_list, dim=1)
        alpha = torch.stack(alpha_list, dim=1).squeeze(-1)

        return z, alpha

    def classify_from_evidence(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes logits from unnormalized gated evidence:
            u = sum_t alpha_t z_t
        """
        u = (alpha.unsqueeze(-1) * z).sum(dim=1)
        logits = self.classifier(u)
        return logits

    def classify_from_masked_evidence(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes logits after deleting evidence terms:
            u_R = sum_{t not in R} alpha_t z_t
        """
        masked_alpha = alpha * keep_mask
        u = (masked_alpha.unsqueeze(-1) * z).sum(dim=1)
        logits = self.classifier(u)
        return logits

    def forward(self, x: torch.Tensor):
        z, alpha = self.encode(x)
        logits = self.classify_from_evidence(z, alpha)

        return logits, {
            "z": z,
            "alpha": alpha,
            "A": self.A,
        }


# ============================================================
# Training and evaluation
# ============================================================

def evaluate_loader(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits, _ = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y_batch.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    dataset_name: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses: List[float] = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits, _ = model(X_batch)
            loss = loss_fn(logits, y_batch)

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss detected on {dataset_name}, seed {cfg.seed}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.clip_grad_max_norm,
            )
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader)
        val_metric = val_metrics["macro_f1"]

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{dataset_name}][{cfg.model_name}][seed={cfg.seed}] "
            f"Epoch {epoch + 1:03d} | "
            f"TrainLoss={np.mean(epoch_losses):.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"ValMacroF1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= cfg.patience:
            print(f"[{dataset_name}][{cfg.model_name}][seed={cfg.seed}] Early stopping triggered.")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
    }

    return model, info


# ============================================================
# Evidence profile extraction
# ============================================================

def get_top_indices(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """
    scores shape: (seq_len,)
    returns top-k indices using largest signed score values.
    """
    seq_len = scores.shape[0]
    k = int(round(fraction * seq_len))
    k = max(1, min(k, seq_len))
    return torch.topk(scores, k=k, largest=True).indices


def make_keep_mask_for_single(seq_len: int, delete_idx: torch.Tensor, device_: torch.device) -> torch.Tensor:
    keep_mask = torch.ones(1, seq_len, device=device_)
    keep_mask[0, delete_idx] = 0.0
    return keep_mask


def compute_classifier_aware_scores(
    model: ProposedUnnormalizedSSM,
    z: torch.Tensor,
    alpha: torch.Tensor,
    logits: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Computes temporal scores for a single batch.

    Args:
        z:      shape (B, T, D)
        alpha:  shape (B, T)
        logits: shape (B, C)

    Returns:
        Dictionary of score tensors, each shape (B, T).
    """
    evidence = alpha.unsqueeze(-1) * z

    pred = logits.argmax(dim=1)

    masked_logits = logits.clone()
    masked_logits[torch.arange(len(pred), device=logits.device), pred] = -float("inf")
    competitor = masked_logits.argmax(dim=1)

    W = model.classifier.weight
    W_pred = W[pred]
    W_comp = W[competitor]

    evidence_norm = torch.linalg.vector_norm(evidence, ord=2, dim=-1)
    gate_alpha = alpha
    class_logit = torch.sum(evidence * W_pred.unsqueeze(1), dim=2)
    margin_logit = torch.sum(evidence * (W_pred - W_comp).unsqueeze(1), dim=2)

    return {
        "evidence_norm": evidence_norm,
        "gate_alpha": gate_alpha,
        "class_logit": class_logit,
        "margin_logit": margin_logit,
    }


def get_selection_score(scores: Dict[str, torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
    if cfg.selection_score not in scores:
        valid = ", ".join(scores.keys())
        raise ValueError(f"Unknown selection_score={cfg.selection_score}. Valid options: {valid}.")
    return scores[cfg.selection_score]


def collect_candidate_profiles(
    model: ProposedUnnormalizedSSM,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    cfg: TrainConfig,
) -> List[Dict]:
    """
    Collects correctly classified test examples and computes classifier-aware evidence profiles.
    """
    model.eval()
    candidates: List[Dict] = []

    with torch.no_grad():
        for idx in range(len(X_test)):
            x = X_test[idx:idx + 1]
            y = int(y_test[idx].item())

            z, alpha = model.encode(x)
            logits = model.classify_from_evidence(z, alpha)
            prob = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(prob, dim=1).item())
            conf = float(prob[0, pred].detach().cpu())

            # Focus on correctly classified examples for qualitative interpretation.
            if pred != y:
                continue

            scores = compute_classifier_aware_scores(model, z, alpha, logits)
            selection_score = get_selection_score(scores, cfg)[0]
            top_idx = get_top_indices(selection_score, cfg.delete_fraction)

            keep_mask = make_keep_mask_for_single(
                seq_len=selection_score.shape[0],
                delete_idx=top_idx,
                device_=x.device,
            )

            deleted_logits = model.classify_from_masked_evidence(z, alpha, keep_mask)
            deleted_prob = torch.softmax(deleted_logits, dim=1)
            deleted_pred = int(torch.argmax(deleted_prob, dim=1).item())
            deleted_conf_original_pred = float(deleted_prob[0, pred].detach().cpu())
            deleted_conf_new_pred = float(deleted_prob[0, deleted_pred].detach().cpu())

            candidates.append({
                "dataset": dataset_name,
                "index": idx,
                "x": x[0].detach().cpu().numpy(),
                "true_label": y,
                "pred_label": pred,
                "clean_conf": conf,
                "deleted_pred_label": deleted_pred,
                "deleted_conf_original_pred": deleted_conf_original_pred,
                "deleted_conf_new_pred": deleted_conf_new_pred,
                "confidence_drop": conf - deleted_conf_original_pred,
                "selection_score": cfg.selection_score,
                "alpha": scores["gate_alpha"][0].detach().cpu().numpy(),
                "evidence_norm": scores["evidence_norm"][0].detach().cpu().numpy(),
                "class_logit": scores["class_logit"][0].detach().cpu().numpy(),
                "margin_logit": scores["margin_logit"][0].detach().cpu().numpy(),
                "top_idx": top_idx.detach().cpu().numpy(),
            })

    return candidates


def select_profiles(candidates: List[Dict], cfg: TrainConfig) -> List[Dict]:
    if not candidates:
        return []

    if cfg.selection_rule == "largest_confidence_drop":
        candidates = sorted(candidates, key=lambda d: d["confidence_drop"], reverse=True)
    elif cfg.selection_rule == "highest_confidence":
        candidates = sorted(candidates, key=lambda d: d["clean_conf"], reverse=True)
    else:
        raise ValueError(f"Unknown selection_rule: {cfg.selection_rule}")

    return candidates[:cfg.n_examples_per_dataset]


# ============================================================
# Plotting
# ============================================================

def normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)

    if not np.isfinite(v_min) or not np.isfinite(v_max) or abs(v_max - v_min) < 1e-12:
        return np.zeros_like(values)

    return (values - v_min) / (v_max - v_min)


def contiguous_regions(indices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Converts sorted indices to contiguous [start, end] regions.
    """
    if len(indices) == 0:
        return []

    indices = np.sort(indices)
    regions = []
    start = int(indices[0])
    prev = int(indices[0])

    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            regions.append((start, prev))
            start = idx
            prev = idx

    regions.append((start, prev))
    return regions


def plot_profile(profile: Dict, output_dir: str, cfg: TrainConfig) -> Tuple[str, str]:
    """
    Saves one qualitative classifier-aware evidence-profile figure as PDF and PNG.

    Figure panels:
        1. Input sequence
        2. Class-logit temporal evidence score
        3. Margin-logit temporal evidence score

    Highlighted regions correspond to the top fraction selected by cfg.selection_score.
    """
    import matplotlib.pyplot as plt

    x = profile["x"]
    class_logit = normalize_to_unit_interval(profile["class_logit"])
    margin_logit = normalize_to_unit_interval(profile["margin_logit"])
    top_idx = profile["top_idx"]
    time_axis = np.arange(len(x))

    regions = contiguous_regions(top_idx)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    # Panel 1: input sequence.
    axes[0].plot(time_axis, x)
    axes[0].set_ylabel("Input")
    axes[0].set_title(
        f"{profile['dataset']} | test index {profile['index']} | "
        f"true={profile['true_label']}, pred={profile['pred_label']}"
    )

    # Panel 2: class-logit score.
    axes[1].plot(time_axis, class_logit)
    if profile["selection_score"] == "class_logit":
        axes[1].scatter(top_idx, class_logit[top_idx], s=18)
    axes[1].set_ylabel("Class-logit")
    axes[1].set_title(r"Classifier-aware score: $w_{\hat c}^{\top} e_t$")

    # Panel 3: margin-logit score.
    axes[2].plot(time_axis, margin_logit)
    if profile["selection_score"] == "margin_logit":
        axes[2].scatter(top_idx, margin_logit[top_idx], s=18)
    axes[2].set_ylabel("Margin-logit")
    axes[2].set_xlabel("Time index")
    axes[2].set_title(r"Classifier-aware score: $(w_{\hat c}-w_{c'})^{\top} e_t$")

    # Shade selected classifier-aware evidence regions in all panels.
    for ax in axes:
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.18)

    fig.text(
        0.01,
        0.01,
        (
            f"Top {cfg.delete_fraction:.0%} selected by {profile['selection_score']}; "
            f"clean confidence={profile['clean_conf']:.3f}; "
            f"confidence after deletion={profile['deleted_conf_original_pred']:.3f}; "
            f"drop={profile['confidence_drop']:.3f}; "
            f"deleted prediction={profile['deleted_pred_label']} "
            f"(confidence={profile['deleted_conf_new_pred']:.3f})"
        ),
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    safe_dataset = str(profile["dataset"]).replace("/", "_")
    base_name = (
        f"classifier_aware_profile_{safe_dataset}"
        f"_idx{profile['index']}"
        f"_{profile['selection_score']}"
        f"_top{int(cfg.top_fraction * 100)}"
    )

    pdf_path = Path(output_dir) / f"{base_name}.pdf"
    png_path = Path(output_dir) / f"{base_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(pdf_path), str(png_path)


# ============================================================
# Main experiment
# ============================================================

def run_one_dataset(cfg: TrainConfig, dataset_name: str) -> List[Dict]:
    print("\n" + "=" * 80)
    print(f"Classifier-aware qualitative evidence profiles | dataset={dataset_name} | seed={cfg.seed}")
    print("=" * 80)

    set_global_seed(cfg.seed)

    X_train, y_train_oh, X_test, y_test_oh = load_dataset(dataset_name)
    y_train = get_integer_labels_from_onehot(y_train_oh)
    y_test = get_integer_labels_from_onehot(y_test_oh)

    n_classes = int(torch.max(y_train).item() + 1)
    input_dim = 1

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        use_validation_split=cfg.use_validation_split,
        val_size=cfg.val_size,
    )

    set_global_seed(cfg.seed)

    model = ProposedUnnormalizedSSM(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        n_classes=n_classes,
    ).to(device)

    model, train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        dataset_name=dataset_name,
    )

    test_metrics = evaluate_loader(model, test_loader)
    print(
        f"[{dataset_name}] Test Acc={test_metrics['accuracy']:.4f} | "
        f"Test MacroF1={test_metrics['macro_f1']:.4f} | "
        f"best_epoch={train_info['best_epoch']} | "
        f"n_params={count_parameters(model)}"
    )

    candidates = collect_candidate_profiles(
        model=model,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
        cfg=cfg,
    )

    selected = select_profiles(candidates, cfg)
    if not selected:
        print(f"[{dataset_name}] No correctly classified candidates found.")
        return []

    for profile in selected:
        pdf_path, png_path = plot_profile(profile, cfg.output_dir, cfg)
        print(
            f"[{dataset_name}] saved profile idx={profile['index']} | "
            f"selection_score={profile['selection_score']} | "
            f"confidence_drop={profile['confidence_drop']:.4f} | "
            f"PDF={pdf_path} | PNG={png_path}"
        )

    return selected


def write_selected_profiles_csv(selected_profiles: List[Dict], output_dir: str) -> str:
    csv_path = Path(output_dir) / "selected_classifier_aware_profiles.csv"

    columns = [
        "dataset",
        "index",
        "true_label",
        "pred_label",
        "clean_conf",
        "deleted_pred_label",
        "deleted_conf_original_pred",
        "deleted_conf_new_pred",
        "confidence_drop",
        "selection_score",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = __import__("csv").DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for p in selected_profiles:
            writer.writerow({col: p.get(col, "") for col in columns})

    return str(csv_path)


def main() -> None:
    cfg = TrainConfig()
    cfg.output_dir = make_output_dir()

    print("RUNNING CLASSIFIER-AWARE QUALITATIVE EVIDENCE-PROFILE EXPERIMENT")
    print("Datasets:", cfg.dataset_names)
    print("Seed:", cfg.seed)
    print("Model:", cfg.model_name)
    print("Selection score:", cfg.selection_score)
    print("Top fraction highlighted:", cfg.top_fraction)
    print("Delete fraction for confidence drop:", cfg.delete_fraction)
    print("Examples per dataset:", cfg.n_examples_per_dataset)
    print("Selection rule:", cfg.selection_rule)
    print("Validation split:", cfg.use_validation_split, "| val_size:", cfg.val_size)
    print("Device:", device)
    print("Output directory:", cfg.output_dir)

    all_selected: List[Dict] = []

    for dataset_name in cfg.dataset_names:
        try:
            selected = run_one_dataset(cfg, dataset_name)
            all_selected.extend(selected)
        except Exception as e:
            print(f"[{dataset_name}] FAILED: {e}")

    csv_path = write_selected_profiles_csv(all_selected, cfg.output_dir)

    print("\nClassifier-aware qualitative evidence-profile experiment finished.")
    print("Output directory:", cfg.output_dir)
    print("Selected profiles CSV:", csv_path)


if __name__ == "__main__":
    main()
