"""Signal complexity measures for neural recordings.

Implements Lempel-Ziv complexity and Hurst exponent (R/S analysis)
for characterizing temporal structure in spike-binned MEA data.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EpochData:
    """A single neural recording epoch with metadata."""
    signal: np.ndarray      # (channels, samples)
    condition: str          # e.g. "rest", "meditation"
    subject_id: str
    epoch_idx: int


def compute_lz_complexity(signal: np.ndarray) -> float:
    """Lempel-Ziv complexity of a 1D signal.

    Binarize (above/below median), then count distinct subsequences
    normalized by signal length.

    Reference: Lempel & Ziv (1976).
    """
    median = np.median(signal)
    binary = "".join("1" if x > median else "0" for x in signal)

    n = len(binary)
    i, k, l = 0, 1, 1
    c = 1

    while k + l <= n:
        if binary[k + l - 1] == binary[i + l - 1]:
            l += 1
        else:
            if l > k - i:
                k = k + l
            else:
                k += 1
            i = 0
            l = 1
            c += 1

    c_norm = c * np.log2(n) / n if n > 0 else 0
    return c_norm


def compute_hurst_exponent(signal: np.ndarray) -> float:
    """Hurst exponent via R/S analysis.

    H > 0.5: persistent (long-range positive correlations)
    H = 0.5: random walk
    H < 0.5: anti-persistent
    """
    n = len(signal)
    if n < 20:
        return 0.5

    max_k = int(np.floor(n / 2))
    rs_list = []
    n_list = []

    for k in [int(2**i) for i in range(2, int(np.log2(max_k)) + 1)]:
        rs_values = []
        for start in range(0, n - k + 1, k):
            segment = signal[start:start + k]
            mean = np.mean(segment)
            deviate = np.cumsum(segment - mean)
            r = np.max(deviate) - np.min(deviate)
            s = np.std(segment, ddof=1) if np.std(segment, ddof=1) > 0 else 1e-10
            rs_values.append(r / s)
        if rs_values:
            rs_list.append(np.mean(rs_values))
            n_list.append(k)

    if len(rs_list) < 2:
        return 0.5

    log_n = np.log(n_list)
    log_rs = np.log(rs_list)
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return slope
