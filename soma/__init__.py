"""SOMA — Self-Organized MEA Architecture.

Self-supervised learning on organoid MEA spike data using
Vision Transformer + JEPA + Barlow Twins.

Discovers discrete network states in biological neural networks
without labels, supervision, or prior assumptions about state structure.
"""

__version__ = "0.1.0"

from .model import BrainJEPA, train_brain_jepa
from .data import load_organoid_data, OrganoidDataset
from .complexity import compute_lz_complexity, compute_hurst_exponent
