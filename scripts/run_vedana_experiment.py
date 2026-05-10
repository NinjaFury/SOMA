#!/usr/bin/env python3
"""Vedanā Gate A/B experiment — gated vs ungated SOMA comparison.

Trains baseline SOMA and Vedanā-gated SOMA on identical data with the
same random seed, then compares silhouette scores and state structure.

Usage:
    python scripts/run_vedana_experiment.py --data path/to/spikes.csv
    python scripts/run_vedana_experiment.py --data spikes.csv --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soma.model import train_brain_jepa
from soma.vedana import train_vedana_jepa
from soma.data import load_organoid_data, summarize_organoid_dataset


def cluster_and_score(embeddings: np.ndarray, k_range: range = range(2, 10)):
    """Cluster embeddings and find optimal k."""
    n = min(50, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(embeddings)

    scores = {}
    for k in k_range:
        if k >= len(reduced):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)
        scores[k] = float(silhouette_score(reduced, labels))

    optimal_k = max(scores, key=scores.get)
    return {
        "optimal_k": optimal_k,
        "best_silhouette": scores[optimal_k],
        "silhouette_by_k": scores,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }


def run(args):
    t_start = time.time()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  Vedanā Gate A/B Experiment")
    print("  Baseline SOMA  vs  Vedanā-gated SOMA")
    print("=" * 60)

    # Load data once
    print("\n1. Loading organoid data...")
    ds = load_organoid_data(
        data_path=Path(args.data),
        bin_sec=args.bin_sec,
        max_windows=args.max_windows,
    )
    print(f"\n{summarize_organoid_dataset(ds)}")
    signals = ds.get_signals()
    spike_rates = ds.get_spike_rates()
    day_labels = ds.get_day_labels()
    out_dir = Path(args.output) if args.output else None

    # --- A: Baseline SOMA ---
    print("\n" + "=" * 60)
    print("  A: Baseline SOMA (no gate)")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    t_a = time.time()
    _, emb_baseline = train_brain_jepa(
        data=signals,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        device=args.device,
        checkpoint_dir=str(out_dir / "baseline" / "ckpt") if out_dir else None,
    )
    t_a = time.time() - t_a

    baseline_results = cluster_and_score(emb_baseline)
    baseline_results["training_time_s"] = t_a

    sr_corr, _ = stats.spearmanr(spike_rates, np.linalg.norm(emb_baseline, axis=1))
    baseline_results["spike_rate_corr"] = float(sr_corr)

    print(f"\n  Baseline: k={baseline_results['optimal_k']}, "
          f"sil={baseline_results['best_silhouette']:.4f}, "
          f"sr_corr={sr_corr:.3f}")

    # --- B: Vedanā-gated SOMA ---
    print("\n" + "=" * 60)
    print("  B: Vedanā-gated SOMA")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    t_b = time.time()
    vedana_model, emb_vedana = train_vedana_jepa(
        data=signals,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        gate_reduction=args.gate_reduction,
        device=args.device,
        checkpoint_dir=str(out_dir / "vedana" / "ckpt") if out_dir else None,
    )
    t_b = time.time() - t_b

    vedana_results = cluster_and_score(emb_vedana)
    vedana_results["training_time_s"] = t_b

    sr_corr_v, _ = stats.spearmanr(
        spike_rates, np.linalg.norm(emb_vedana, axis=1))
    vedana_results["spike_rate_corr"] = float(sr_corr_v)

    # Gate analysis
    x_tensor = torch.from_numpy(signals).float().unsqueeze(1)
    gate_scores_all = []
    with torch.no_grad():
        device = next(vedana_model.parameters()).device
        for i in range(0, len(signals), args.batch_size):
            batch = x_tensor[i:i + args.batch_size].to(device)
            gs = vedana_model.get_gate_scores(batch)
            gate_scores_all.append(gs)
    gate_scores_all = np.concatenate(gate_scores_all, axis=0)

    vedana_results["gate_mean"] = float(gate_scores_all.mean())
    vedana_results["gate_std"] = float(gate_scores_all.std())
    vedana_results["gate_sparsity"] = float((gate_scores_all < 0.5).mean())
    vedana_results["gate_per_patch_mean"] = gate_scores_all.mean(axis=0).tolist()

    print(f"\n  Vedanā: k={vedana_results['optimal_k']}, "
          f"sil={vedana_results['best_silhouette']:.4f}, "
          f"sr_corr={sr_corr_v:.3f}")
    print(f"  Gate: mean={vedana_results['gate_mean']:.3f}, "
          f"std={vedana_results['gate_std']:.3f}, "
          f"sparsity={vedana_results['gate_sparsity']:.3f}")

    # --- Comparison ---
    sil_delta = (vedana_results["best_silhouette"]
                 - baseline_results["best_silhouette"])
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Silhouette:  baseline={baseline_results['best_silhouette']:.4f}"
          f"  vedanā={vedana_results['best_silhouette']:.4f}"
          f"  delta={sil_delta:+.4f}")
    print(f"  Optimal k:   baseline={baseline_results['optimal_k']}"
          f"  vedanā={vedana_results['optimal_k']}")
    print(f"  SR corr:     baseline={baseline_results['spike_rate_corr']:.3f}"
          f"  vedanā={vedana_results['spike_rate_corr']:.3f}")
    print(f"  Time:        baseline={t_a:.1f}s  vedanā={t_b:.1f}s")

    verdict = "IMPROVED" if sil_delta > 0.01 else (
        "DEGRADED" if sil_delta < -0.01 else "NEUTRAL")
    print(f"\n  Verdict: {verdict} (delta={sil_delta:+.4f})")

    report = {
        "experiment": "vedana_gate_ab",
        "seed": seed,
        "config": {
            "n_windows": ds.n_epochs,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "epochs": args.epochs,
            "gate_reduction": args.gate_reduction,
        },
        "baseline": baseline_results,
        "vedana": vedana_results,
        "comparison": {
            "silhouette_delta": sil_delta,
            "verdict": verdict,
        },
        "duration_seconds": time.time() - t_start,
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "vedana_ab_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        np.save(str(out_dir / "baseline_embeddings.npy"), emb_baseline)
        np.save(str(out_dir / "vedana_embeddings.npy"), emb_vedana)
        np.save(str(out_dir / "gate_scores.npy"), gate_scores_all)
        print(f"\n  Results saved to {out_dir}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vedanā Gate A/B: gated vs ungated SOMA")
    parser.add_argument("--data", required=True, help="Path to spike CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bin-sec", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--gate-reduction", type=int, default=4,
                        help="Gate hidden dim = embed_dim / reduction")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/vedana_ab")
    args = parser.parse_args()

    run(args)
