#!/usr/bin/env python3
"""Run SOMA experiment on organoid MEA spike data.

Usage:
    python scripts/run_experiment.py --data path/to/SpikeDataToShare_fs437data.csv
    python scripts/run_experiment.py --data data.csv --epochs 50 --embed-dim 256 --depth 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soma.model import BrainJEPA, train_brain_jepa
from soma.data import load_organoid_data, summarize_organoid_dataset
from soma.complexity import compute_lz_complexity, compute_hurst_exponent


def run(args):
    t_start = time.time()

    print("=" * 60)
    print("  SOMA: Self-Organized MEA Architecture")
    print("  Organoid MEA spike data -> self-supervised embeddings")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading organoid spike data...")
    ds = load_organoid_data(
        data_path=Path(args.data),
        bin_sec=args.bin_sec,
        max_windows=args.max_windows,
    )
    print(f"\n{summarize_organoid_dataset(ds)}")

    # 2. Train SOMA
    print("\n2. Training SOMA...")
    signals = ds.get_signals()
    checkpoint_dir = str(Path(args.output) / "checkpoints") if args.output else None

    model, embeddings = train_brain_jepa(
        data=signals,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
    )
    print(f"   Embeddings: {embeddings.shape}")

    # 3. Evaluate
    print("\n3. Evaluating hypotheses...")
    results = {}
    day_labels = ds.get_day_labels()
    spike_rates = ds.get_spike_rates()

    # Network State Discovery
    print("\n   Network State Discovery")
    n_components = min(50, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    scores = {}
    for k in range(2, 10):
        if k >= len(reduced):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels)
        scores[k] = float(sil)
        print(f"     k={k}: silhouette={sil:.4f}")

    optimal_k = max(scores, key=scores.get)
    results["optimal_k"] = optimal_k
    results["best_silhouette"] = scores[optimal_k]
    results["silhouette_by_k"] = scores

    # Developmental Trajectory
    print("\n   Developmental Trajectory")
    unique_days = sorted(set(day_labels))
    if len(unique_days) >= 2:
        sil = silhouette_score(embeddings, day_labels)
        results["day_silhouette"] = float(sil)
        print(f"     Day clustering silhouette: {sil:.3f}")

    # Spike rate correlation
    emb_norms = np.linalg.norm(embeddings, axis=1)
    sr_corr, sr_p = stats.spearmanr(spike_rates, emb_norms)
    results["spike_rate_corr"] = float(sr_corr)
    print(f"     Spike rate <-> embedding norm: r={sr_corr:.3f}")

    elapsed = time.time() - t_start

    report = {
        "experiment": "SOMA",
        "duration_seconds": elapsed,
        "config": {
            "n_windows": ds.n_epochs,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "epochs": args.epochs,
        },
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"  Optimal k={optimal_k}, silhouette={scores[optimal_k]:.4f}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"{'='*60}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        np.save(str(out / "embeddings.npy"), embeddings)
        np.save(str(out / "day_labels.npy"), day_labels)
        print(f"\nResults saved to {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOMA experiment")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to FinalSpark spike CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bin-sec", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/soma")
    args = parser.parse_args()

    run(args)
