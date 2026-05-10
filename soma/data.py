"""Organoid MEA spike data pipeline.

Loads FinalSpark CSV spike recordings, bins into activity matrices,
and creates sliding windows for SOMA training.

Pipeline:
  Spike CSV -> per-electrode binning -> (C, T) activity matrices
  -> sliding windows -> OrganoidDataset
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .complexity import compute_lz_complexity, compute_hurst_exponent


# --- Data classes ---

@dataclass
class OrganoidEpoch:
    """Single activity window from organoid recording."""
    signal: np.ndarray          # (n_electrodes, n_bins)
    day: int                    # recording day index
    segment_idx: int            # segment within day
    window_idx: int             # window within segment
    spike_rate: float           # mean spikes per bin
    burst_index: float = 0.0    # burstiness metric
    network_sync: float = 0.0   # cross-electrode synchrony
    hurst: float = -1.0         # Hurst exponent (-1 = not computed)


@dataclass
class OrganoidDataset:
    """Collection of organoid epochs ready for SOMA training."""
    epochs: list[OrganoidEpoch]
    metadata: dict = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def days_covered(self) -> list[int]:
        return sorted(set(e.day for e in self.epochs))

    def get_signals(self) -> np.ndarray:
        """(N, C, T) array for model input."""
        return np.stack([e.signal for e in self.epochs])

    def get_day_labels(self) -> np.ndarray:
        return np.array([e.day for e in self.epochs])

    def get_spike_rates(self) -> np.ndarray:
        return np.array([e.spike_rate for e in self.epochs])

    def get_hurst_values(self) -> np.ndarray:
        return np.array([e.hurst for e in self.epochs])


# --- Spike binning ---

def load_organoid_data(
    data_path: Path,
    bin_sec: float = 0.1,
    window_bins: int = 10,
    stride_bins: int = 5,
    max_windows: int | None = None,
    compute_complexity: bool = False,
) -> OrganoidDataset:
    """Load FinalSpark CSV and create binned activity windows.

    Args:
        data_path: Path to SpikeDataToShare_fs437data.csv
        bin_sec: Bin width in seconds
        window_bins: Number of bins per window (T dimension)
        stride_bins: Stride between windows
        max_windows: Cap total windows (for development)
        compute_complexity: Whether to compute Hurst per segment

    Returns:
        OrganoidDataset ready for SOMA training.
    """
    _log(f"Loading spike data from {data_path}")
    df = pd.read_csv(data_path)

    # Normalize column names
    if '_time' in df.columns:
        df = df.rename(columns={'_time': 'time', '_value': 'amplitude', 'index': 'electrode'})

    # Convert datetime to seconds
    if pd.api.types.is_string_dtype(df['time']):
        dt = pd.to_datetime(df['time'], utc=True)
        df['time'] = (dt - dt.min()).dt.total_seconds()

    # Identify recording days by gaps > 1 hour
    time_sorted = df['time'].sort_values().values
    day_breaks = np.where(np.diff(time_sorted) > 3600)[0]
    day_starts = [time_sorted[0]] + [time_sorted[i + 1] for i in day_breaks]
    day_ends = [time_sorted[i] for i in day_breaks] + [time_sorted[-1]]

    n_electrodes = int(df['electrode'].max()) + 1
    _log(f"  {len(df):,} spikes, {n_electrodes} electrodes, {len(day_starts)} days")

    all_epochs: list[OrganoidEpoch] = []

    for day_idx, (ds, de) in enumerate(zip(day_starts, day_ends)):
        day_df = df[(df['time'] >= ds) & (df['time'] <= de)]
        if len(day_df) < 100:
            continue

        t_min = day_df['time'].min()
        duration = day_df['time'].max() - t_min
        n_bins = int(duration / bin_sec) + 1

        # Bin spikes into (electrodes, time_bins) matrix
        matrix = np.zeros((n_electrodes, n_bins), dtype=np.float32)
        bin_idx = ((day_df['time'].values - t_min) / bin_sec).astype(int)
        elec_idx = day_df['electrode'].values.astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n_bins) & (elec_idx >= 0) & (elec_idx < n_electrodes)
        np.add.at(matrix, (elec_idx[valid], bin_idx[valid]), 1)

        # Compute segment-level Hurst if requested
        segment_hurst = -1.0
        if compute_complexity and n_bins >= 500:
            network_sum = matrix.sum(axis=0)
            segment_hurst = compute_hurst_exponent(network_sum[:500])

        # Sliding windows
        n_windows = (n_bins - window_bins) // stride_bins + 1
        for w in range(n_windows):
            start = w * stride_bins
            window = matrix[:, start:start + window_bins]
            sr = float(window.sum()) / (n_electrodes * window_bins)

            if window.sum() == 0:
                continue

            all_epochs.append(OrganoidEpoch(
                signal=window,
                day=day_idx,
                segment_idx=day_idx,
                window_idx=w,
                spike_rate=sr,
                burst_index=compute_burst_index(window),
                network_sync=compute_network_synchrony(window),
                hurst=segment_hurst,
            ))

        _log(f"  Day {day_idx}: {duration/3600:.1f}h, {n_bins} bins, "
             f"{n_windows} windows, {len(day_df):,} spikes")

    # Cap if needed
    if max_windows and len(all_epochs) > max_windows:
        indices = np.linspace(0, len(all_epochs) - 1, max_windows, dtype=int)
        all_epochs = [all_epochs[i] for i in indices]

    _log(f"  Total: {len(all_epochs)} windows across {len(set(e.day for e in all_epochs))} days")

    return OrganoidDataset(
        epochs=all_epochs,
        metadata={
            "source": str(data_path),
            "n_electrodes": n_electrodes,
            "bin_sec": bin_sec,
            "window_bins": window_bins,
            "stride_bins": stride_bins,
            "n_days": len(day_starts),
            "total_spikes": len(df),
        },
    )


def summarize_organoid_dataset(ds: OrganoidDataset) -> str:
    """Human-readable dataset summary."""
    rates = ds.get_spike_rates()
    days = ds.get_day_labels()
    lines = [
        f"OrganoidDataset: {ds.n_epochs} windows",
        f"  Days: {sorted(set(days))}",
        f"  Windows per day: {dict(zip(*np.unique(days, return_counts=True)))}",
        f"  Spike rate: {rates.mean():.4f} +/- {rates.std():.4f}",
    ]
    return "\n".join(lines)


# --- Feature computation ---

def compute_burst_index(window: np.ndarray) -> float:
    """Burstiness: ratio of peak to mean firing rate."""
    rates = window.sum(axis=0)
    mean_rate = rates.mean()
    if mean_rate < 1e-8:
        return 0.0
    return float(rates.max() / mean_rate)


def compute_network_synchrony(window: np.ndarray) -> float:
    """Cross-electrode synchrony via mean pairwise correlation."""
    active = window[window.sum(axis=1) > 0]
    if len(active) < 2:
        return 0.0
    stds = active.std(axis=1)
    valid = stds > 1e-8
    if valid.sum() < 2:
        return 0.0
    corr = np.corrcoef(active[valid])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(np.mean(corr[mask]))


def detect_spikes(signal: np.ndarray, threshold_std: float = 3.0) -> np.ndarray:
    """Simple threshold-based spike detection."""
    std = np.std(signal)
    if std < 1e-8:
        return np.array([])
    threshold = threshold_std * std
    crossings = np.where(np.abs(signal) > threshold)[0]
    if len(crossings) == 0:
        return crossings
    # Merge crossings within 10 samples
    gaps = np.diff(crossings)
    keep = np.concatenate([[True], gaps > 10])
    return crossings[keep]


def _log(msg: str):
    print(f"[soma:data] {msg}", file=sys.stderr)
