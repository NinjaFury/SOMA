"""SOMA Model — Self-Organized MEA Architecture.

Self-contained PyTorch implementation: ViT encoder + JEPA predictor + Barlow Twins.

Architecture:
  Input: [B, 1, C, T] (1-channel C-electrode x T-timeframe neural recording)
  PatchEmbed: Conv2d -> patch tokens of embed_dim
  Context Encoder (ViT): visible patches -> context embeddings
  Target Encoder: EMA copy of context encoder (no gradient)
  Predictor: small transformer mapping context -> predicted target embeddings
  Anti-collapse: Barlow Twins loss (cross-correlation -> identity matrix)

Combined loss:
  L = L_jepa + alpha * L_barlow
  L_jepa   = SmoothL1(predicted, target) for masked patches
  L_barlow = sum((diag - 1)^2) + lambda * sum(off_diag^2)
"""
from __future__ import annotations

import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Core building blocks ---

class PatchEmbed(nn.Module):
    """2D patch embedding via Conv2d."""

    def __init__(self, img_size=(64, 160), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (1, patch_size)  # patch along time axis only
        self.num_patches_2d = (img_size[0] // self.patch_size[0],
                               img_size[1] // self.patch_size[1])
        self.num_patches = self.num_patches_2d[0] * self.num_patches_2d[1]
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """Feed-forward with GELU."""

    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Encoder ---

class EEGEncoder(nn.Module):
    """ViT encoder for neural recordings. Input [B, 1, C, T] -> [B, N, embed_dim]."""

    def __init__(self, n_channels=64, n_frames=160, patch_size=16,
                 embed_dim=768, depth=6, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed((n_channels, n_frames), patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Sinusoidal 2D positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.pos_embed.data.copy_(self._sincos_2d(embed_dim, self.patch_embed.num_patches_2d))

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _sincos_2d(self, dim, grid_size):
        gh = np.arange(grid_size[0], dtype=float)
        gw = np.arange(grid_size[1], dtype=float)
        grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, -1)

        def _sin1d(d, pos):
            omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=float) / (d / 2))
            out = np.einsum('m,d->md', pos.ravel(), omega)
            return np.concatenate([np.sin(out), np.cos(out)], axis=1)

        emb = np.concatenate([_sin1d(dim // 2, grid[1]), _sin1d(dim // 2, grid[0])], axis=1)
        return torch.from_numpy(emb).float().unsqueeze(0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def embed(self, x):
        """Global embedding: mean-pool patch tokens -> (B, embed_dim)."""
        return self.forward(x).mean(dim=1)


# --- Predictor ---

class EEGPredictor(nn.Module):
    """Predicts target patch embeddings from context + mask tokens."""

    def __init__(self, embed_dim=768, pred_dim=384, depth=4, num_heads=12, num_patches=640):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, pred_dim), requires_grad=False)
        pos = self._sincos_1d(pred_dim, num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        self.blocks = nn.ModuleList([Block(pred_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, embed_dim)

    def _sincos_1d(self, dim, length):
        pos = np.arange(length, dtype=float)
        omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=float) / (dim / 2))
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    def forward(self, ctx_emb, ctx_idx, tgt_idx):
        B = ctx_emb.shape[0]
        x = self.proj_in(ctx_emb)

        ctx_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1,
                               ctx_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = x + ctx_pos

        tgt_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1,
                               tgt_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        masks = self.mask_token.expand(B, tgt_idx.shape[1], -1) + tgt_pos

        x = torch.cat([x, masks], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj_out(x[:, -tgt_idx.shape[1]:])


# --- Barlow Twins Anti-Collapse ---

def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.005) -> torch.Tensor:
    """Barlow Twins: cross-correlation matrix -> identity target.

    From Horace Barlow's 1961 redundancy reduction hypothesis.
    Each embedding dimension captures unique, non-redundant information.
    """
    z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-5)
    z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-5)

    N = z_a.shape[0]
    c = (z_a.T @ z_b) / N

    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c ** 2).sum() - (torch.diagonal(c) ** 2).sum()
    return on_diag + lambd * off_diag


# --- Full SOMA System ---

class BrainJEPA(nn.Module):
    """SOMA: Self-Organized MEA Architecture.

    Complete JEPA system: context encoder + target encoder (EMA) + predictor.

    Usage:
        model = BrainJEPA(n_channels=32, n_frames=10, patch_size=10)
        result = model.train_step(batch_x)   # {'loss', 'jepa_loss', 'barlow_loss'}
        embeddings = model.embed(batch_x)     # (B, embed_dim)
    """

    def __init__(self, n_channels=64, n_frames=160, patch_size=16,
                 embed_dim=768, depth=6, num_heads=12,
                 pred_dim=None, pred_depth=4,
                 mask_ratio=0.75, ema_decay=0.996,
                 barlow_alpha=0.1, barlow_lambda=0.005):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.barlow_alpha = barlow_alpha
        self.barlow_lambda = barlow_lambda

        # Auto-adjust heads to divide evenly
        while embed_dim % num_heads != 0:
            num_heads -= 1

        pred_dim = pred_dim or embed_dim // 2
        pred_heads = num_heads
        while pred_dim % pred_heads != 0:
            pred_heads -= 1

        self.context_encoder = EEGEncoder(n_channels, n_frames, patch_size,
                                          embed_dim, depth, num_heads)
        num_patches = self.context_encoder.patch_embed.num_patches

        # Target encoder (EMA copy, no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = EEGPredictor(embed_dim, pred_dim, pred_depth,
                                      pred_heads, num_patches)

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.criterion = nn.SmoothL1Loss()

    @torch.no_grad()
    def _update_target(self):
        """Exponential moving average update of target encoder."""
        for p_c, p_t in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1 - self.ema_decay)

    def _create_masks(self, batch_size, device):
        n_mask = int(self.num_patches * self.mask_ratio)
        n_visible = self.num_patches - n_mask
        ctx, tgt = [], []
        for _ in range(batch_size):
            perm = torch.randperm(self.num_patches, device=device)
            ctx.append(perm[:n_visible])
            tgt.append(perm[n_visible:])
        return torch.stack(ctx), torch.stack(tgt)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One training step. Returns dict with 'loss', 'jepa_loss', 'barlow_loss'.

        Args:
            x: [B, 1, C, T] neural recording tensor
        """
        B = x.shape[0]
        device = x.device
        ctx_idx, tgt_idx = self._create_masks(B, device)

        # Target encoder (no grad)
        with torch.no_grad():
            tgt_all = self.target_encoder(x)
            target_emb = torch.gather(tgt_all, 1,
                                      tgt_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Context encoder on visible patches only
        ctx_patches = self.context_encoder.patch_embed(x) + self.context_encoder.pos_embed
        ctx_emb = torch.gather(ctx_patches, 1,
                               ctx_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        for blk in self.context_encoder.blocks:
            ctx_emb = blk(ctx_emb)
        ctx_emb = self.context_encoder.norm(ctx_emb)

        # Predictor: context -> predicted target
        predicted = self.predictor(ctx_emb, ctx_idx, tgt_idx)

        # JEPA loss
        jepa_loss = self.criterion(predicted, target_emb)

        # Barlow Twins anti-collapse on global embeddings
        ctx_global = ctx_emb.mean(dim=1)
        tgt_global = tgt_all.mean(dim=1)
        bt_loss = barlow_twins_loss(ctx_global, tgt_global, self.barlow_lambda)

        loss = jepa_loss + self.barlow_alpha * bt_loss

        self._update_target()

        return {"loss": loss, "jepa_loss": jepa_loss, "barlow_loss": bt_loss}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global embeddings from context encoder. (B, embed_dim)."""
        return self.context_encoder.embed(x)

    @torch.no_grad()
    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch embeddings. (B, num_patches, embed_dim)."""
        return self.context_encoder(x)


# --- Training Loop ---

def train_brain_jepa(
    data: np.ndarray,
    labels: np.ndarray | None = None,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1.5e-4,
    embed_dim: int = 768,
    depth: int = 6,
    warmup_epochs: int = 10,
    device: str | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[BrainJEPA, np.ndarray]:
    """Train SOMA and return model + embeddings.

    Args:
        data: (N, C, T) neural recording array
        labels: optional (N,) array (not used in training, only logging)
        n_epochs: training epochs
        batch_size: per-step batch size
        lr: peak learning rate
        embed_dim: embedding dimension
        depth: transformer depth
        warmup_epochs: linear warmup epochs
        device: 'cuda', 'cpu', or auto-detect
        checkpoint_dir: save checkpoints here

    Returns:
        (trained_model, embeddings) where embeddings is (N, embed_dim).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = data.shape[0]
    _log(f"Training SOMA: {N} windows of [{data.shape[1]}x{data.shape[2]}] on {device}")
    _log(f"  embed_dim={embed_dim}, depth={depth}, epochs={n_epochs}, batch={batch_size}")

    # Auto-select patch_size to divide time dimension evenly
    n_frames = data.shape[2]
    patch_size = 16
    if n_frames < patch_size:
        for ps in [n_frames, n_frames // 2, 5, 2, 1]:
            if ps > 0 and n_frames % ps == 0:
                patch_size = ps
                break

    model = BrainJEPA(
        n_channels=data.shape[1], n_frames=n_frames,
        patch_size=patch_size,
        embed_dim=embed_dim, depth=depth,
    ).to(device)
    _log(f"  patch_size={patch_size} -> {model.num_patches} patches")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"  Trainable params: {n_params / 1e6:.1f}M")

    x_tensor = torch.from_numpy(data).float().unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True)

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

    best_loss = float("inf")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            result = model.train_step(batch_x)
            optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += result["loss"].item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log(f"  Epoch {epoch+1:3d}/{n_epochs} | loss={avg:.5f} | "
                 f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if avg < best_loss:
            best_loss = avg
            if checkpoint_dir:
                from pathlib import Path
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           str(Path(checkpoint_dir) / "soma_best.pt"))

    # Extract embeddings
    model.eval()
    _log("Extracting embeddings...")
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
    print(f"[soma] {msg}", file=sys.stderr)
