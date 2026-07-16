from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

from recognition.inference.daily30_sentence_feature_utils import (
    blank_context_sequence,
    boundary_shift_sequence,
    partial_observation_sequence,
    sliding_window_sequences,
)
from recognition.inference.daily30_sentence_model_utils import BiGRUSentenceClassifier


class SeqDataset(Dataset):
    def __init__(self, split_csv: Path, feature_dir: Path, split: str, label_to_idx: Dict[str, int]):
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
        if missing > 0:
            print(f"WARN: split={split} missing feature files skipped: {missing}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        stem = Path(row["video_name"]).stem
        npz = np.load(self.feature_dir / f"{stem}.npz", allow_pickle=True)
        x = npz["feature"].astype(np.float32)
        y = self.label_to_idx[row["template_id"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate daily30 fixed-sentence BiGRU.")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--out-metrics", required=True)
    ap.add_argument("--out-confusion", required=True)
    ap.add_argument("--out-predictions", required=True)
    ap.add_argument("--out-proxy-metrics")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pooling", default="mean", choices=["mean", "mean_max"])
    ap.add_argument("--split", default="test")
    return ap.parse_args()


def predict_probs(model: BiGRUSentenceClassifier, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None, ...].astype(np.float32)))
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def summarize_scores(y_true: list[int], probs: list[np.ndarray]) -> dict:
    pred = [int(np.argmax(p)) for p in probs]
    return {
        "top1": float(accuracy_score(y_true, pred)) if y_true else 0.0,
        "top3": float(top_k_accuracy_score(y_true, np.array(probs), k=min(3, len(probs[0])))) if y_true else 0.0,
        "n_samples": len(y_true),
    }


def build_proxy_metrics(model: BiGRUSentenceClassifier, ds: SeqDataset, idx_to_label: Dict[int, str]) -> dict:
    scenarios = {
        "partial_50": lambda x: [partial_observation_sequence(x, 0.50)],
        "partial_75": lambda x: [partial_observation_sequence(x, 0.75)],
        "boundary_shift_left_10": lambda x: [boundary_shift_sequence(x, -0.10)],
        "boundary_shift_right_10": lambda x: [boundary_shift_sequence(x, 0.10)],
        "blank_context_10_10": lambda x: [blank_context_sequence(x, 0.10, 0.10)],
        "sliding_window_context_25": lambda x: sliding_window_sequences(x, context_ratio=0.25, stride_ratio=0.1),
    }

    scenario_probs: dict[str, list[np.ndarray]] = {name: [] for name in scenarios}
    y_true: list[int] = []
    sliding_hits_top1 = 0
    sliding_hits_top3 = 0
    per_sample_sliding_top1_hit: list[int] = []

    for i in range(len(ds)):
        x_t, y_t, meta = ds[i]
        x = x_t.numpy()
        y = int(y_t.item())
        y_true.append(y)
        for name, builder in scenarios.items():
            variants = builder(x)
            variant_probs = [predict_probs(model, variant) for variant in variants]
            if name.startswith("sliding_window"):
                best_idx = int(np.argmax([float(np.max(p)) for p in variant_probs]))
                chosen = variant_probs[best_idx]
                tops = [set(np.argsort(-p)[: min(3, len(p))].tolist()) for p in variant_probs]
                preds = [int(np.argmax(p)) for p in variant_probs]
                top1_hit = int(y in preds)
                top3_hit = int(any(y in top for top in tops))
                sliding_hits_top1 += top1_hit
                sliding_hits_top3 += top3_hit
                per_sample_sliding_top1_hit.append(top1_hit)
            else:
                chosen = variant_probs[0]
            scenario_probs[name].append(chosen)

    metrics = {
        "split": ds.items[0]["split"] if ds.items else "unknown",
        "n_samples": len(y_true),
        "scenarios": {},
    }
    top1_values = []
    top3_values = []
    for name, probs in scenario_probs.items():
        if name.startswith("sliding_window"):
            top1 = float(sliding_hits_top1 / len(y_true)) if y_true else 0.0
            top3 = float(sliding_hits_top3 / len(y_true)) if y_true else 0.0
            metrics["scenarios"][name] = {
                "top1_hit_rate": top1,
                "top3_hit_rate": top3,
                "n_samples": len(y_true),
            }
        else:
            summary = summarize_scores(y_true, probs)
            metrics["scenarios"][name] = summary
            top1 = summary["top1"]
            top3 = summary["top3"]
        top1_values.append(top1)
        top3_values.append(top3)

    metrics["proxy_summary"] = {
        "mean_top1": float(np.mean(top1_values)) if top1_values else 0.0,
        "mean_top3": float(np.mean(top3_values)) if top3_values else 0.0,
        "min_top1": float(np.min(top1_values)) if top1_values else 0.0,
        "min_top3": float(np.min(top3_values)) if top3_values else 0.0,
    }
    signer_stats = defaultdict(lambda: {"n": 0, "partial_50_correct": 0, "sliding_window_top1_hit": 0})
    for i, row in enumerate(ds.items):
        signer = row["signer_id"]
        signer_stats[signer]["n"] += 1
        partial_pred = int(np.argmax(scenario_probs["partial_50"][i]))
        signer_stats[signer]["partial_50_correct"] += int(partial_pred == y_true[i])
        signer_stats[signer]["sliding_window_top1_hit"] += per_sample_sliding_top1_hit[i]
    metrics["per_signer_proxy"] = {
        signer: {
            "partial_50_top1": stats["partial_50_correct"] / stats["n"] if stats["n"] else 0.0,
            "sliding_window_top1_hit_rate": stats["sliding_window_top1_hit"] / stats["n"] if stats["n"] else 0.0,
            "n_samples": stats["n"],
        }
        for signer, stats in sorted(signer_stats.items())
    }
    metrics["label_space"] = [idx_to_label[i] for i in range(len(idx_to_label))]
    return metrics


def main() -> None:
    args = parse_args()
    lm = json.loads(Path(args.label_map).read_text(encoding="utf-8"))
    label_to_idx = {k: int(v) for k, v in lm["label_to_idx"].items()}
    idx_to_label = {int(k): v for k, v in lm["idx_to_label"].items()}

    ds = SeqDataset(Path(args.split_csv), Path(args.feature_dir), args.split, label_to_idx)
    if len(ds) == 0:
        raise RuntimeError(f"No rows found for split={args.split}")
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    x0, _, _ = ds[0]
    model = BiGRUSentenceClassifier(
        int(x0.shape[-1]),
        args.hidden_size,
        args.num_layers,
        args.dropout,
        len(label_to_idx),
        pooling=args.pooling,
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    ys, ps, probs = [], [], []
    pred_rows: List[Dict[str, str]] = []

    with torch.no_grad():
        for x, y, meta in loader:
            logits = model(x)
            p = torch.softmax(logits, dim=-1).numpy()
            pred_idx = np.argmax(p, axis=1)
            ys.extend(y.numpy().tolist())
            ps.extend(pred_idx.tolist())
            probs.extend(p.tolist())
            top3_idx = np.argsort(-p, axis=1)[:, :3]
            for i in range(len(pred_idx)):
                mi = {k: meta[k][i] for k in meta}
                pred_rows.append(
                    {
                        "video_name": mi["video_name"],
                        "split": mi["split"],
                        "sequence_id": mi["sequence_id"],
                        "sentence_text": mi["sentence_text"],
                        "signer_id": mi["signer_id"],
                        "y_true": idx_to_label[int(y.numpy()[i])],
                        "y_pred": idx_to_label[int(pred_idx[i])],
                        "confidence": float(p[i][pred_idx[i]]),
                        "top3": "|".join(idx_to_label[int(t)] for t in top3_idx[i]),
                    }
                )

    top1 = accuracy_score(ys, ps)
    top3 = top_k_accuracy_score(ys, np.array(probs), k=min(3, len(label_to_idx)))

    per_template = defaultdict(lambda: {"n": 0, "correct": 0})
    for yt, yp in zip(ys, ps):
        key = idx_to_label[int(yt)]
        per_template[key]["n"] += 1
        per_template[key]["correct"] += int(yt == yp)
    per_template_acc = {k: (v["correct"] / v["n"] if v["n"] else 0.0) for k, v in sorted(per_template.items())}

    signer_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in pred_rows:
        signer_stats[row["signer_id"]]["n"] += 1
        signer_stats[row["signer_id"]]["correct"] += int(row["y_true"] == row["y_pred"])
    per_signer_acc = {k: (v["correct"] / v["n"] if v["n"] else 0.0) for k, v in sorted(signer_stats.items())}

    metrics = {
        "split": args.split,
        "sentence_top1": top1,
        "sentence_top3": top3,
        "per_template_accuracy": per_template_acc,
        "per_signer_accuracy": per_signer_acc,
        "n_samples": len(ys),
    }
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = [idx_to_label[i] for i in range(len(label_to_idx))]
    cm = confusion_matrix(ys, ps, labels=list(range(len(label_to_idx))))
    with Path(args.out_confusion).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([labels[i]] + row)

    with Path(args.out_predictions).open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["video_name", "split", "sequence_id", "sentence_text", "signer_id", "y_true", "y_pred", "confidence", "top3"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    proxy_metrics = build_proxy_metrics(model, ds, idx_to_label)
    proxy_path = Path(args.out_proxy_metrics) if args.out_proxy_metrics else Path(args.out_metrics).with_name("metrics_realtime_proxy.json")
    proxy_path.write_text(json.dumps(proxy_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
