from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

DATASET_MAIN_FOLDER = "Datasets"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_dataset_files(
    root_folder: str | Path,
    dataset_name: str,
    extensions: Tuple[str, ...] = (".txt", ""),
) -> Tuple[Path, Path]:
    """
    Find TRAIN and TEST files for a dataset.

    Supported layouts:
    1) Flat:
       Datasets/ECG5000_TRAIN
       Datasets/ECG5000_TEST

    2) Nested:
       Datasets/ECG5000/ECG5000_TRAIN
       Datasets/ECG5000/ECG5000_TEST
    """
    root = Path(root_folder)

    candidate_dirs = [
        root,
        root / dataset_name,
    ]

    train_base = f"{dataset_name}_TRAIN"
    test_base = f"{dataset_name}_TEST"

    for folder in candidate_dirs:
        for ext in extensions:
            train_file = folder / f"{train_base}{ext}"
            test_file = folder / f"{test_base}{ext}"
            if train_file.exists() and test_file.exists():
                return train_file, test_file

    searched = []
    for folder in candidate_dirs:
        for ext in extensions:
            searched.append(str(folder / f"{train_base}{ext}"))
            searched.append(str(folder / f"{test_base}{ext}"))

    raise FileNotFoundError(
        f"Dataset files for '{dataset_name}' were not found.\n"
        f"Searched paths:\n- " + "\n- ".join(searched)
    )


def normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalize each time series independently:
        x <- (x - mean) / std
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape}.")

    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    std[std == 0] = 1e-8
    return (features - mean) / std


def load_dataset(
    dataset_name: str,
    dataset_main_folder: str | Path = DATASET_MAIN_FOLDER,
    return_label_encoder: bool = False,
):
    train_file, test_file = _find_dataset_files(dataset_main_folder, dataset_name)

    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for dataset '{dataset_name}', "
            f"got {train_data.shape} and {test_data.shape}."
        )

    X_train = train_data[:, 1:]
    X_test = test_data[:, 1:]

    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError(f"NaN values found in features for dataset '{dataset_name}'.")

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    all_labels = np.concatenate([train_data[:, 0], test_data[:, 0]]).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(all_labels)

    y_train = enc.transform(train_data[:, 0].reshape(-1, 1))
    y_test = enc.transform(test_data[:, 0].reshape(-1, 1))

    outputs = (
        torch.from_numpy(X_train).float().to(device),
        torch.from_numpy(y_train).int().to(device),
        torch.from_numpy(X_test).float().to(device),
        torch.from_numpy(y_test).int().to(device),
    )

    if not return_label_encoder:
        return outputs

    label_info: Dict[str, object] = {
        "categories": enc.categories_[0].tolist(),
        "n_classes": len(enc.categories_[0]),
    }
    return (*outputs, label_info)


def get_integer_labels_from_onehot(onehot_labels: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot labels to integer class indices.
    """
    if onehot_labels.ndim != 2:
        raise ValueError(
            f"Expected 2D one-hot label tensor, got shape {tuple(onehot_labels.shape)}."
        )
    return torch.argmax(onehot_labels, dim=1)


if __name__ == "__main__":
    example_dataset = "ECG5000"
    try:
        X_train, y_train, X_test, y_test, label_info = load_dataset(
            example_dataset,
            return_label_encoder=True,
        )
        print(f"Loaded dataset: {example_dataset}")
        print("X_train:", tuple(X_train.shape))
        print("y_train:", tuple(y_train.shape))
        print("X_test :", tuple(X_test.shape))
        print("y_test :", tuple(y_test.shape))
        print("Label info:", label_info)
        print("Device:", device)
    except Exception as e:
        print("Sanity check failed:", e)