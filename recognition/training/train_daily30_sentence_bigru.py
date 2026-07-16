from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier, augment_feature_sequence


class SeqDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        feature_dir: Path,
        split: str,
        label_to_idx: Dict[str, int],
        noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
        time_mask_width: int = 0,
        seed: int = 42,
    ):
        rows = list(csv.DictReader(split_csv.open("r", encoding="utf-8-sig", newline="")))
        self.items = []
        missing = 0
        for row in rows:
            if row["split"] != split:
                continue
            stem = Path(row["video_name"]).stem
            if not (feature_dir / f"{stem}.npz").exists():
                missing += 1
                continue
            self.items.append(row)
        self.feature_dir = feature_dir
        self.label_to_idx = label_to_idx
        self.enable_augment = split == "train"
        self.noise_std = float(noise_std)
        self.frame_dropout_prob = float(frame_dropout_prob)
        self.time_mask_width = int(time_mask_width)
        self.rng = np.random.default_rng(seed)
        if missing > 0:
            print(f"WARN: split={split} missing feature files skipped: {missing}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        stem = Path(row["video_name"]).stem
        npz = np.load(self.feature_dir / f"{stem}.npz", allow_pickle=True)
        x = npz["feature"].astype(np.float32)
        if self.enable_augment:
            x = augment_feature_sequence(
                x,
                noise_std=self.noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
                time_mask_width=self.time_mask_width,
                rng=self.rng,
            )
        y = self.label_to_idx[row["template_id"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), row["video_name"]


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True, grad_clip: float | None = None):
    model.train(train)
    losses = []
    ys, ps, probs = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(float(loss.item()))
        p = torch.softmax(logits, dim=-1)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(torch.argmax(p, dim=-1).detach().cpu().numpy().tolist())
        probs.extend(p.detach().cpu().numpy().tolist())

    top1 = accuracy_score(ys, ps) if ys else 0.0
    top3 = top_k_accuracy_score(ys, np.array(probs), k=min(3, len(probs[0]))) if ys else 0.0
    return {"loss": float(np.mean(losses)) if losses else math.inf, "top1": float(top1), "top3": float(top3)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train daily30 fixed-sentence BiGRU.")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--scheduler-patience", type=int, default=4)
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--frame-dropout-prob", type=float, default=0.0)
    ap.add_argument("--time-mask-width", type=int, default=0)
    ap.add_argument("--pooling", default="mean", choices=["mean", "mean_max"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss(label_smoothing: float) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = list(csv.DictReader(Path(args.split_csv).open("r", encoding="utf-8-sig", newline="")))
    templates = sorted({row["template_id"] for row in rows})
    label_to_idx = {k: i for i, k in enumerate(templates)}
    idx_to_label = {i: k for k, i in label_to_idx.items()}

    train_ds = SeqDataset(
        Path(args.split_csv),
        Path(args.feature_dir),
        "train",
        label_to_idx,
        noise_std=args.noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
        time_mask_width=args.time_mask_width,
        seed=args.seed,
    )
    dev_ds = SeqDataset(Path(args.split_csv), Path(args.feature_dir), "dev", label_to_idx)
    if len(train_ds) == 0 or len(dev_ds) == 0:
        raise RuntimeError("Empty train/dev dataset. Ensure features are extracted and split is valid.")

    x0, _, _ = train_ds[0]
    input_dim = int(x0.shape[-1])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    model = BiGRUSentenceClassifier(
        input_dim,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        len(templates),
        pooling=args.pooling,
    ).to(device)
    criterion = build_loss(args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = -1.0
    best_epoch = -1
    patience_left = args.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True, grad_clip=args.grad_clip)
        dv = run_epoch(model, dev_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step(dv["loss"])
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_top1": tr["top1"],
            "train_top3": tr["top3"],
            "dev_loss": dv["loss"],
            "dev_top1": dv["top1"],
            "dev_top3": dv["top3"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if dv["top1"] > best_top1:
            best_top1 = dv["top1"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    metrics = {
        "best_dev_top1": best_top1,
        "best_epoch": best_epoch,
        "num_templates": len(templates),
        "num_train_samples": len(train_ds),
        "num_dev_samples": len(dev_ds),
        "device": str(device),
        "pooling": args.pooling,
        "noise_std": args.noise_std,
        "frame_dropout_prob": args.frame_dropout_prob,
        "time_mask_width": args.time_mask_width,
    }
    (out_dir / "label_map_v1.json").write_text(
        json.dumps({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "train_history_v1.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_summary_v1.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
