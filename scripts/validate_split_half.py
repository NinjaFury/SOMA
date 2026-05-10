#!/usr/bin/env python3
"""Split-half validation of network state discovery.

Proves SOMA's network state discovery generalizes by:
1. Splitting the recording into interleaved 1-hour blocks (A/B)
2. Training SOMA independently on each half
3. Comparing discovered state structure between halves

If both halves discover similar states with comparable silhouette
scores, the finding is robust — not a statistical artifact.

Usage:
    python scripts/validate_split_half.py --data path/to/spikes.csv
    python scripts/validate_split_half.py --data spikes.csv --embed-dim 128 --depth 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soma.model import BrainJEPA


# --- Config ---
BIN_SEC = 0.1
WINDOW_BINS = 10
STRIDE_BINS = 5


def load_and_split(data_path: str, max_windows: int = 25000):
    """Load spike data and split into interleaved 1-hour halves."""
    print(f"Loading {data_path}")
    df = pd.read_csv(data_path)

    if '_time' in df.columns:
        df = df.rename(columns={'_time': 'time', '_value': 'amplitude', 'index': 'electrode'})

    if pd.api.types.is_string_dtype(df['time']):
        dt = pd.to_datetime(df['time'], utc=True)
        df['time'] = (dt - dt.min()).dt.total_seconds()

    t_min = df['time'].min()
    block_id = ((df['time'] - t_min) / 3600).astype(int)
    df_a = df[block_id % 2 == 0].copy()
    df_b = df[block_id % 2 == 1].copy()

    print(f"  Half A: {len(df_a):,} spikes | Half B: {len(df_b):,} spikes")

    windows_a = _spikes_to_windows(df_a, max_windows, "A")
    windows_b = _spikes_to_windows(df_b, max_windows, "B")
    return windows_a, windows_b


def _spikes_to_windows(df, max_windows, label):
    n_electrodes = 32
    t_min, t_max = df['time'].min(), df['time'].max()
    n_bins = int((t_max - t_min) / BIN_SEC) + 1

    matrix = np.zeros((n_electrodes, n_bins), dtype=np.float32)
    bin_idx = ((df['time'].values - t_min) / BIN_SEC).astype(int)
    elec_idx = df['electrode'].values.astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (elec_idx >= 0) & (elec_idx < n_electrodes)
    np.add.at(matrix, (elec_idx[valid], bin_idx[valid]), 1)

    n_windows = (n_bins - WINDOW_BINS) // STRIDE_BINS + 1
    if n_windows > max_windows:
        indices = np.linspace(0, n_windows - 1, max_windows, dtype=int)
    else:
        indices = np.arange(n_windows)

    windows = []
    for i in indices:
        start = i * STRIDE_BINS
        w = matrix[:, start:start + WINDOW_BINS]
        if w.sum() > 0:
            windows.append(w)

    windows = np.array(windows, dtype=np.float32)
    print(f"  [{label}] {len(windows):,} windows")
    return windows


def train_and_embed(windows, label, embed_dim, depth, epochs, batch_size, device):
    """Train SOMA and extract embeddings."""
    print(f"\n  [{label}] Training: embed_dim={embed_dim}, depth={depth}, epochs={epochs}")

    n_ch, n_t = windows.shape[1], windows.shape[2]
    tensor = torch.from_numpy(windows).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BrainJEPA(
        n_channels=n_ch, n_frames=n_t, patch_size=n_t,
        embed_dim=embed_dim, depth=depth, num_heads=8, pred_depth=3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n = 0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            result = model.train_step(batch)
            optimizer.zero_grad()
            result['loss'].backward()
            optimizer.step()
            model._update_target()
            total_loss += result['loss'].item()
            n += 1
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [{label}] Epoch {epoch:3d}/{epochs} | loss={total_loss/n:.5f}")

    model.eval()
    embs = []
    with torch.no_grad():
        for (batch,) in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
            embs.append(model.embed(batch.to(device)).cpu().numpy())
    return np.concatenate(embs)


def find_states(embeddings, label):
    """Sweep k=2..11 and find optimal clusters."""
    print(f"\n  [{label}] Finding states...")
    n = len(embeddings)
    emb = embeddings[np.random.choice(n, min(n, 10000), replace=False)] if n > 10000 else embeddings

    best_k, best_sil, scores = 2, -1, {}
    for k in range(2, 12):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        sil = silhouette_score(emb, km.fit_predict(emb))
        scores[k] = sil
        if sil > best_sil:
            best_sil, best_k = sil, k
        print(f"  [{label}] k={k}: sil={sil:.4f}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    counts = [int((labels == i).sum()) for i in range(best_k)]
    print(f"  [{label}] Optimal: k={best_k}, sil={best_sil:.4f}, counts={counts}")

    return {
        'optimal_k': best_k, 'best_silhouette': best_sil,
        'silhouette_by_k': {str(k): v for k, v in scores.items()},
        'state_counts': counts,
    }


def compare(a, b):
    """Compare state structures between halves."""
    k_diff = abs(a['optimal_k'] - b['optimal_k'])
    sil_diff = abs(a['best_silhouette'] - b['best_silhouette'])
    both_good = (a['optimal_k'] >= 2 and b['optimal_k'] >= 2
                 and a['best_silhouette'] > 0.3 and b['best_silhouette'] > 0.3)

    ks = sorted(set(a['silhouette_by_k'].keys()) & set(b['silhouette_by_k'].keys()))
    curve_a = [a['silhouette_by_k'][k] for k in ks]
    curve_b = [b['silhouette_by_k'][k] for k in ks]
    curve_corr = float(np.corrcoef(curve_a, curve_b)[0, 1])

    prop_a = max(a['state_counts']) / sum(a['state_counts'])
    prop_b = max(b['state_counts']) / sum(b['state_counts'])

    validated = (both_good and k_diff <= 2 and sil_diff < 0.15 and curve_corr > 0.5)

    print(f"\n{'='*60}")
    print(f"  Half A: k={a['optimal_k']}, sil={a['best_silhouette']:.4f}")
    print(f"  Half B: k={b['optimal_k']}, sil={b['best_silhouette']:.4f}")
    print(f"  k diff={k_diff}, sil diff={sil_diff:.4f}, curve r={curve_corr:.4f}")
    print(f"  VALIDATED: {validated}")
    print(f"{'='*60}")

    return {
        'k_a': a['optimal_k'], 'k_b': b['optimal_k'],
        'sil_a': a['best_silhouette'], 'sil_b': b['best_silhouette'],
        'curve_correlation': curve_corr,
        'prop_a': prop_a, 'prop_b': prop_b,
        'validated': validated,
    }


def main():
    parser = argparse.ArgumentParser(description="Split-half validation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=25000)
    parser.add_argument("--output", type=str, default="results/split_half")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.time()

    windows_a, windows_b = load_and_split(args.data, args.max_windows)
    emb_a = train_and_embed(windows_a, "A", args.embed_dim, args.depth,
                            args.epochs, args.batch_size, device)
    emb_b = train_and_embed(windows_b, "B", args.embed_dim, args.depth,
                            args.epochs, args.batch_size, device)
    states_a = find_states(emb_a, "A")
    states_b = find_states(emb_b, "B")
    comparison = compare(states_a, states_b)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        'experiment': 'split_half_validation',
        'duration': time.time() - start,
        'config': vars(args),
        'half_a': states_a, 'half_b': states_b,
        'comparison': comparison,
    }
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    np.save(out / 'embeddings_a.npy', emb_a)
    np.save(out / 'embeddings_b.npy', emb_b)
    print(f"\nSaved to {out} ({time.time()-start:.0f}s)")


if __name__ == '__main__':
    main()
