"""Vedanā Gate — mandatory valence-scoring layer for SOMA.

Buddhist origin: Vedanā (feeling-tone) is the pre-cognitive classification
that arises upon contact with any sense impression — pleasant, unpleasant,
or neutral — before conceptual processing begins. It is the second link
in paṭicca samuppāda that shapes all downstream perception.

Architecture: a lightweight per-patch scorer inserted between patch embedding
and the transformer encoder. Each patch gets an independent valence score
(no cross-patch interaction). This is contact-level, not cognitive-level.

The gate lives only in the context encoder path. The target encoder sees
ungated patches — so the gate must learn which patches carry the signal
needed to predict masked representations. That IS vedanā: the pre-cognitive
assessment of what matters.

Combined with SOMA:
  MEA spikes → bin → patches → **Vedanā Gate** → ViT → JEPA → states
"""
from __future__ import annotations

import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .model import BrainJEPA, EEGEncoder, EEGPredictor, barlow_twins_loss


class VedanaGate(nn.Module):
    """Per-patch valence scoring. Pre-cognitive, independent per patch.

    Each patch embedding is scored through a small MLP → sigmoid → [0, 1].
    Output is the input scaled by the gate: x * gate_score.

    Initialized near pass-through so training starts from SOMA baseline.
    The JEPA loss teaches the gate what to keep vs suppress.
    """

    def __init__(self, embed_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(embed_dim // reduction, 16)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Initialize near pass-through: sigmoid(2.0) ≈ 0.88
        nn.init.zeros_(self.scorer[2].weight)
        nn.init.constant_(self.scorer[2].bias, 2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate patch embeddings.

        Args:
            x: (B, N, D) patch embeddings

        Returns:
            gated: (B, N, D) — x scaled by gate scores
            scores: (B, N, 1) — raw gate values in [0, 1]
        """
        scores = torch.sigmoid(self.scorer(x))  # (B, N, 1)
        return x * scores, scores


class VedanaBrainJEPA(BrainJEPA):
    """SOMA with Vedana Gate — valence-scored self-supervised learning.

    Identical to BrainJEPA except:
    1. A VedanaGate is applied to context patches after masking
    2. Target encoder path is ungated (asymmetric by design)
    3. Gate statistics are tracked in train_step output

    Usage:
        model = VedanaBrainJEPA(n_channels=32, n_frames=10, patch_size=10)
        result = model.train_step(batch_x)
        # result includes 'gate_mean', 'gate_std', 'gate_sparsity'
    """

    def __init__(self, *args, gate_reduction: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.vedana_gate = VedanaGate(self.embed_dim, reduction=gate_reduction)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training step with vedana gating on context path.

        Gate is applied after patch selection (masking), before transformer.
        Target path remains ungated — the asymmetry is the learning signal.
        """
        B = x.shape[0]
        device = x.device
        ctx_idx, tgt_idx = self._create_masks(B, device)

        # Target encoder (no grad, ungated)
        with torch.no_grad():
            tgt_all = self.target_encoder(x)
            target_emb = torch.gather(
                tgt_all, 1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Context encoder: patch → select visible → GATE → transformer
        ctx_patches = (self.context_encoder.patch_embed(x)
                       + self.context_encoder.pos_embed)
        ctx_emb = torch.gather(
            ctx_patches, 1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # --- Vedana Gate ---
        ctx_emb, gate_scores = self.vedana_gate(ctx_emb)

        for blk in self.context_encoder.blocks:
            ctx_emb = blk(ctx_emb)
        ctx_emb = self.context_encoder.norm(ctx_emb)

        # Predictor
        predicted = self.predictor(ctx_emb, ctx_idx, tgt_idx)

        # Losses
        jepa_loss = self.criterion(predicted, target_emb)

        ctx_global = ctx_emb.mean(dim=1)
        tgt_global = tgt_all.mean(dim=1)
        bt_loss = barlow_twins_loss(ctx_global, tgt_global, self.barlow_lambda)

        loss = jepa_loss + self.barlow_alpha * bt_loss

        self._update_target()

        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "barlow_loss": bt_loss,
            "gate_mean": gate_scores.mean().detach(),
            "gate_std": gate_scores.std().detach(),
            "gate_sparsity": (gate_scores < 0.5).float().mean().detach(),
        }

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract gated global embeddings. (B, embed_dim)."""
        patches = (self.context_encoder.patch_embed(x)
                   + self.context_encoder.pos_embed)
        gated, _ = self.vedana_gate(patches)
        for blk in self.context_encoder.blocks:
            gated = blk(gated)
        return self.context_encoder.norm(gated).mean(dim=1)

    @torch.no_grad()
    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract gated per-patch embeddings. (B, N, D)."""
        patches = (self.context_encoder.patch_embed(x)
                   + self.context_encoder.pos_embed)
        gated, _ = self.vedana_gate(patches)
        for blk in self.context_encoder.blocks:
            gated = blk(gated)
        return self.context_encoder.norm(gated)

    @torch.no_grad()
    def get_gate_scores(self, x: torch.Tensor) -> np.ndarray:
        """Get vedana scores for all patches. (B, N) numpy array."""
        patches = (self.context_encoder.patch_embed(x)
                   + self.context_encoder.pos_embed)
        _, scores = self.vedana_gate(patches)
        return scores.squeeze(-1).cpu().numpy()


# --- Training Loop ---

def train_vedana_jepa(
    data: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1.5e-4,
    embed_dim: int = 768,
    depth: int = 6,
    gate_reduction: int = 4,
    warmup_epochs: int = 10,
    device: str | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[VedanaBrainJEPA, np.ndarray]:
    """Train Vedana-gated SOMA and return model + embeddings.

    Mirrors train_brain_jepa() with VedanaBrainJEPA + gate logging.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = data.shape[0]
    n_frames = data.shape[2]
    _log(f"Training Vedana-SOMA: {N} windows of "
         f"[{data.shape[1]}x{n_frames}] on {device}")

    # Auto patch_size
    patch_size = 16
    if n_frames < patch_size:
        for ps in [n_frames, n_frames // 2, 5, 2, 1]:
            if ps > 0 and n_frames % ps == 0:
                patch_size = ps
                break

    model = VedanaBrainJEPA(
        n_channels=data.shape[1], n_frames=n_frames,
        patch_size=patch_size,
        embed_dim=embed_dim, depth=depth,
        gate_reduction=gate_reduction,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate_params = sum(p.numel() for p in model.vedana_gate.parameters())
    _log(f"  patch_size={patch_size} -> {model.num_patches} patches")
    _log(f"  Trainable params: {n_params/1e6:.1f}M "
         f"(gate: {gate_params} / {gate_params/n_params*100:.1f}%)")

    x_tensor = torch.from_numpy(data).float().unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.05,
    )

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gate_mean = 0.0
        epoch_gate_sparsity = 0.0
        t0 = time.time()

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            result = model.train_step(batch_x)
            optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += result["loss"].item()
            epoch_gate_mean += result["gate_mean"].item()
            epoch_gate_sparsity += result["gate_sparsity"].item()

        scheduler.step()
        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        avg_gate = epoch_gate_mean / n_batches
        avg_sparsity = epoch_gate_sparsity / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log(f"  Epoch {epoch+1:3d}/{n_epochs} | loss={avg_loss:.5f} | "
                 f"gate={avg_gate:.3f} sparsity={avg_sparsity:.3f} | "
                 f"{elapsed:.1f}s")

        if checkpoint_dir:
            from pathlib import Path as P
            P(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(),
                       str(P(checkpoint_dir) / "vedana_best.pt"))

    # Extract embeddings
    model.eval()
    _log("Extracting gated embeddings...")
    embeddings = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = x_tensor[i:i+batch_size].to(device)
            emb = model.embed(batch)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    _log(f"Done. Embeddings shape: {embeddings.shape}")

    return model, embeddings


def _log(msg: str):
    print(f"[vedana] {msg}", file=sys.stderr)
