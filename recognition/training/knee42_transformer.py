"""Training for the Knee42 Transformer path.

Deliberately imports the feature pipeline and the model from
``recognition.transformer`` rather than restating them.  The upstream training
scripts carried their own copies of ``interp_missing`` / ``resample`` /
``featurize`` / the network definition, which is exactly how a training pipeline
and its deployed counterpart drift apart.  There is one definition here.

Two protocols are supported:

``loso``
    Leave-one-signer-out.  The test signer is excluded from training entirely
    and 15% of the remaining data becomes the validation split.  This is the
    protocol every published accuracy figure comes from.

``final``
    Train on every signer with a random 12% validation split.  This is how the
    released weights were produced.  The resulting validation score is
    **optimistic and not a held-out estimate** -- every signer it validates on
    is also in its training data.  Never publish it as accuracy.

Reproducing the published numbers needs the feature cache, which is not part of
this repository; see the data policy in the README.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recognition.transformer.features import (
    LANDMARK_DIM,
    SEQUENCE_LENGTH,
    featurize,
    interp_missing,
    resample,
)
from recognition.transformer.model import Knee42Transformer


CACHE_VERSION = "knee42_features_upright_v2"


@dataclass
class Dataset:
    sequences: list[np.ndarray]
    labels: np.ndarray
    signers: np.ndarray
    label_ids: list[str]

    def __len__(self) -> int:
        return len(self.sequences)


def load_dataset(data_root: Path | str) -> Dataset:
    """Read ``research_manifest.csv`` and its ``features_final/`` cache.

    Missing coordinates are interpolated once, here, so the per-epoch cost is
    only augmentation and resampling.
    """
    data_root = Path(data_root)
    manifest = data_root / "research_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")

    label_ids = sorted({row["label_id"] for row in rows})
    index_of = {label: index for index, label in enumerate(label_ids)}

    sequences, labels, signers = [], [], []
    for row in rows:
        path = data_root / "features_final" / f"{row['sample_id']}.npz"
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["cache_version"].item())
            if version != CACHE_VERSION:
                raise ValueError(f"{path.name}: cache_version {version!r} != {CACHE_VERSION!r}")
            values = payload["values"].astype(np.float32)
        if values.ndim != 2 or values.shape[1] != LANDMARK_DIM:
            raise ValueError(f"{path.name}: expected [frames,{LANDMARK_DIM}], got {values.shape}")
        sequences.append(interp_missing(values))
        labels.append(index_of[row["label_id"]])
        signers.append(row["signer_id"])

    return Dataset(sequences, np.asarray(labels), np.asarray(signers), label_ids)


def augment(sequence: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Speed warp, boundary crop, frame dropout, then scale/shift/jitter.

    Everything here is temporal or affine: no mirroring.  Mirror augmentation was
    measured and made cross-signer accuracy worse, see docs/evaluation/.
    """
    warped = resample(sequence, max(8, int(round(len(sequence) * rng.uniform(0.8, 1.25)))))

    margin = max(1, int(0.05 * len(warped)))
    start = rng.randint(0, margin)
    end = len(warped) - rng.randint(0, margin)
    cropped = warped[start : max(end, start + 8)]

    keep = np.sort(
        rng.choice(len(cropped), size=max(8, int(len(cropped) * 0.85)), replace=False)
    )
    kept = cropped[keep]

    scaled = kept * rng.uniform(0.9, 1.1)
    scaled = scaled + rng.normal(0, 0.01, size=scaled.shape).astype(np.float32)
    return scaled + rng.uniform(-0.05, 0.05, size=(1, scaled.shape[1])).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
) -> dict[str, float]:
    """Top-1, top-3 and macro top-1. Macro is the selection metric: classes are equal."""
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        chunks.append(model(batch).cpu().numpy())
    logits = np.concatenate(chunks)
    predicted = logits.argmax(1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    present = np.unique(labels)
    return {
        "top1": float((predicted == labels).mean()),
        "top3": float(np.any(top3 == labels[:, None], axis=1).mean()),
        "macro_top1": float(
            np.mean([(predicted[labels == label] == label).mean() for label in present])
        ),
    }


def _train(
    dataset: Dataset,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    rng: np.random.RandomState,
    device: torch.device | str,
    epochs: int,
    patience: int,
    batch_size: int,
    encoder_checkpoint: Path | None,
    learning_rate: float,
    encoder_learning_rate: float,
) -> tuple[nn.Module, float, int]:
    """Shared loop: AdamW + cosine schedule, early stopping on validation macro top-1."""
    model = Knee42Transformer(len(dataset.label_ids))
    if encoder_checkpoint is not None:
        checkpoint = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        pretrained = Knee42Transformer(int(checkpoint["n_classes"]))
        pretrained.load_state_dict(checkpoint["state_dict"], strict=True)
        pretrained.head = nn.Linear(pretrained.head.in_features, len(dataset.label_ids))
        model = pretrained
    model = model.to(device)

    if encoder_checkpoint is not None:
        encoder_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("head.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": encoder_learning_rate},
                {"params": model.head.parameters(), "lr": learning_rate},
            ],
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=encoder_learning_rate, weight_decay=0.01
        )

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    validation_features = np.stack(
        [featurize(dataset.sequences[index]) for index in validation_index]
    )
    validation_labels = dataset.labels[validation_index]

    best_score, best_state, waited, epoch = -1.0, None, 0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_index))
        for start in range(0, len(order), batch_size):
            batch_index = train_index[order[start : start + batch_size]]
            inputs = np.stack(
                [featurize(augment(dataset.sequences[index], rng)) for index in batch_index]
            )
            targets = torch.from_numpy(dataset.labels[batch_index]).to(device)
            optimizer.zero_grad()
            criterion(model(torch.from_numpy(inputs).to(device)), targets).backward()
            optimizer.step()
        schedule.step()

        score = evaluate(model, validation_features, validation_labels, device)["macro_top1"]
        if score > best_score:
            best_score, waited = score, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score, epoch + 1


def train_leave_one_signer_out(
    dataset: Dataset,
    test_signer: str,
    seed: int,
    *,
    device: torch.device | str = "cuda",
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 64,
    encoder_checkpoint: Path | None = None,
) -> tuple[nn.Module, dict]:
    """Train with ``test_signer`` fully excluded and score on that signer."""
    if test_signer not in set(dataset.signers):
        raise ValueError(f"unknown signer {test_signer!r}; have {sorted(set(dataset.signers))}")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    trainable = np.where(dataset.signers != test_signer)[0]
    held_out = np.where(dataset.signers == test_signer)[0]
    rng.shuffle(trainable)
    split = int(len(trainable) * 0.15)
    validation_index, train_index = trainable[:split], trainable[split:]

    model, best_score, epochs_ran = _train(
        dataset,
        train_index,
        validation_index,
        rng=rng,
        device=device,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        encoder_checkpoint=encoder_checkpoint,
        learning_rate=1e-3,
        encoder_learning_rate=3e-4 if encoder_checkpoint is None else 1e-4,
    )
    test_features = np.stack([featurize(dataset.sequences[index]) for index in held_out])
    return model, {
        "protocol": "leave_one_signer_out",
        "test_signer": test_signer,
        "seed": seed,
        "val_macro": best_score,
        "epochs_ran": epochs_ran,
        "test": evaluate(model, test_features, dataset.labels[held_out], device),
    }


def train_final(
    dataset: Dataset,
    seed: int,
    *,
    device: torch.device | str = "cuda",
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 64,
    encoder_checkpoint: Path | None = None,
    validation_fraction: float = 0.12,
) -> tuple[nn.Module, dict]:
    """Train on every signer. The reported validation score is NOT held out."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    order = rng.permutation(len(dataset))
    split = int(len(order) * validation_fraction)
    validation_index, train_index = order[:split], order[split:]

    model, best_score, epochs_ran = _train(
        dataset,
        train_index,
        validation_index,
        rng=rng,
        device=device,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        encoder_checkpoint=encoder_checkpoint,
        learning_rate=1e-3,
        encoder_learning_rate=1e-4,
    )
    return model, {
        "protocol": "all_signers",
        "seed": seed,
        "val_macro_mixed": best_score,
        "epochs_ran": epochs_ran,
        "held_out_test_set": None,
        "warning": (
            "val_macro_mixed is measured on a random split of the training signers "
            "and is optimistic; it is not an accuracy estimate."
        ),
    }


def save_checkpoint(model: nn.Module, dataset: Dataset, path: Path, metrics: dict) -> None:
    """Write a checkpoint in the shape ``recognition.transformer.recognizer`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "n_classes": len(dataset.label_ids),
            "label_ids": list(dataset.label_ids),
            "sequence_length": SEQUENCE_LENGTH,
            "metrics": metrics,
        },
        path,
    )
