# SOMA — Self-Organized MEA Architecture

Self-supervised learning on organoid MEA spike data using Vision Transformer + JEPA + Barlow Twins.

SOMA discovers discrete network states in biological neural networks without labels, supervision, or prior assumptions about state structure.

## Key Results

- **9 discrete network states** discovered in FinalSpark organoid MEA data (silhouette = 0.636)
- **Hierarchical state structure**: 2 → 4 → 9 states emerge across model scales
- **Split-half validated**: 4 independent models (2 CPU + 2 GPU) all converge on the same binary state structure
- **Developmental trajectory**: entropy increases monotonically from 0.511 bits (day 0) to 1.918 bits (day 4)

## Architecture

```
MEA Spikes → Bin (0.1s) → Activity Windows (32 electrodes × 10 bins)
    ↓
Context Encoder (ViT) ← 75% spatiotemporal masking
    ↓                        ↓
Predictor → L_jepa      Target Encoder (EMA)
    ↓                        ↓
         L = L_jepa + α·L_barlow
    ↓
Embeddings → KMeans Clustering → Network States
```

**JEPA**: Predicts target encoder representations (not raw signals) for masked patches. Learns abstract structure.

**Barlow Twins**: Cross-correlation → identity matrix. Prevents embedding collapse, ensures each dimension captures unique information.

## Installation

```bash
git clone https://github.com/NinjaFury/SOMA.git
cd SOMA
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. GPU recommended but not required.

## Usage

### Run experiment

```bash
python scripts/run_experiment.py --data path/to/SpikeDataToShare_fs437data.csv
```

Full options:
```bash
python scripts/run_experiment.py \
    --data spikes.csv \
    --epochs 50 \
    --embed-dim 256 \
    --depth 6 \
    --batch-size 32 \
    --device cuda \
    --output results/soma
```

### Split-half validation

```bash
python scripts/validate_split_half.py --data path/to/spikes.csv
```

### Python API

```python
from soma import BrainJEPA, train_brain_jepa, load_organoid_data

# Load data
ds = load_organoid_data("SpikeDataToShare_fs437data.csv")
signals = ds.get_signals()  # (N, 32, 10)

# Train
model, embeddings = train_brain_jepa(
    data=signals,
    n_epochs=50,
    embed_dim=256,
    depth=6,
    device="cuda",
)

# Cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=9, n_init=10, random_state=42)
states = km.fit_predict(embeddings)
print(f"Silhouette: {silhouette_score(embeddings, states):.3f}")
```

## Vedanā Gate (Experimental)

SOMA v0.2 introduces the **Vedanā Gate** — a mandatory valence-scoring layer inserted between patch embedding and the transformer encoder. Inspired by the Buddhist concept of vedanā (feeling-tone), each patch is independently scored before any cross-patch interaction occurs.

```
Patches (B, N, D)
    ↓
VedanāGate: Linear → GELU → Linear → σ → per-patch score ∈ [0,1]
    ↓
gated = patches * scores
    ↓
Transformer Blocks → JEPA
```

The gate adds 0.4% parameters and learns end-to-end from the JEPA loss — no separate supervision.

### Run A/B experiment

```bash
python scripts/run_vedana_experiment.py --data path/to/spikes.csv
```

### Python API

```python
from soma import VedanaBrainJEPA, train_vedana_jepa

model, embeddings = train_vedana_jepa(
    data=signals,
    n_epochs=50,
    embed_dim=256,
    depth=6,
    device="cuda",
)

# Inspect what the gate learned
gate_scores = model.get_gate_scores(batch_x)  # (B, N) per-patch scores
```

## Data

This work uses spike data from the [FinalSpark](https://finalspark.com/) Neuroplatform — a cloud-accessible platform for biological neural network experiments. The dataset contains MEA recordings from organoid cultures across multiple days.

The raw data is not included in this repository. Contact FinalSpark for access.

## Paper

Preprint: [SOMA: Self-Organized MEA Architecture — Self-Supervised Discovery of Network States in Organoid Neural Cultures](https://doi.org/10.1101/2026.05.09.724050)

```bibtex
@article{pathirana2026soma,
  title={SOMA: Self-Organized MEA Architecture --- Self-Supervised Discovery
         of Network States in Organoid Neural Cultures},
  author={Pathirana, Ishara R.},
  journal={bioRxiv},
  year={2026},
  doi={10.1101/2026.05.09.724050}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

FinalSpark team (Fred Jordan, Martin Kutter, Jean-Marc Comby, Flora Brozzi, Dr. Ewelina Kurtys) for Neuroplatform access and data.
