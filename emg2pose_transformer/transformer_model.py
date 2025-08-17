# transformer_model.py
import torch
import torch.nn as nn
import math
from einops import rearrange

class SinusoidalPositionalEncoding(nn.Module):
    """
    Batch-first sinusoidal positional encoding.
    Input x shape: (B, T, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.dtype)
        return self.dropout(x)


class EMGHandPoseTransformer(nn.Module):
    """
    Sequence-to-sequence Transformer for EMG -> joint angles.
    Designed to keep time resolution (no downsampling inside the model).
    Predicts per-timestep poses: output shape (B, T, pose_dim).
    Also contains simple target-normalization buffers + helper to set them.
    """
    def __init__(self,
                 num_channels: int = 16,
                 window_size: int = 200,       # T (number of time samples per input window)
                 pose_dim: int = 20,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers_temporal: int = 4,
                 num_layers_spatial: int = 2,
                 mlp_hidden_dim: int = 256,
                 dropout: float = 0.1,
                 max_len: int = 200):
        """
        max_len: maximum sequence length for positional encodings (should be >= window_size)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.window_size = window_size
        self.num_channels = num_channels
        self.pose_dim = pose_dim
        self.embed_dim = embed_dim

        # --- Temporal stream (per-time-step embedding of C channels) ---
        # Input x shape: (B, T, C)
        self.temporal_embed = nn.Linear(num_channels, embed_dim)   # maps C -> D per timestep
        self.temporal_pos_encoder = SinusoidalPositionalEncoding(embed_dim, dropout, max_len=max_len)
        temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers_temporal)

        # --- Spatial stream (per-channel embedding of T samples) ---
        # Input for spatial: rearranged to (B, C, T) -> Linear(T -> D)
        self.spatial_embed = nn.Linear(window_size, embed_dim)     # maps time-axis per channel to D
        self.spatial_pos_encoder = nn.Parameter(torch.randn(1, num_channels, embed_dim))
        spatial_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_layers_spatial)

        # --- Fusion: temporal queries attend to spatial keys/values ---
        self.fusion_cross_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            dropout=dropout,
                                                            batch_first=True)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # --- Sequence-wise regression head: (B, T, D) -> (B, T, pose_dim) ---
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, pose_dim)
        )

        # --- Target normalization buffers (set by training script) ---
        # By default mean=0, std=1 (i.e., no-op).
        self.register_buffer('target_mean', torch.zeros(pose_dim))
        self.register_buffer('target_std', torch.ones(pose_dim))

    # helper to set target normalization stats (call from training script)
    def set_target_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """
        mean/std: 1D tensors of length pose_dim. Will be stored as buffers.
        Call before training/validation loops.
        """
        assert mean.shape[0] == self.pose_dim and std.shape[0] == self.pose_dim
        self.target_mean.data.copy_(mean.to(self.target_mean.dtype))
        self.target_std.data.copy_(std.to(self.target_std.dtype))

    def denormalize(self, normalized_pose: torch.Tensor) -> torch.Tensor:
        """
        normalized_pose: (B, T, pose_dim) or (B, pose_dim)
        returns denormalized: same shape
        """
        return normalized_pose * self.target_std.view(1, 1, -1) + self.target_mean.view(1, 1, -1)

    def forward(self, x: torch.Tensor, return_denorm: bool = False):
        """
        Args:
            x: (B, T, C) raw EMG window (already preprocessed: filtering + scaling)
            return_denorm: if True, also return denormalized predictions (in original pose units)
        Returns:
            preds_norm: (B, T, pose_dim) normalized predictions (mean 0, std 1 if stats set)
            if return_denorm=True: also returns preds_denorm (B, T, pose_dim)
        NOTE: This module expects the training script to normalize targets with the same stats used here.
        """
        B, T, C = x.shape
        assert C == self.num_channels, f"Expected input channels={self.num_channels}, got {C}"
        assert T == self.window_size, f"Expected input time length={self.window_size}, got {T}"

        # --- Temporal stream ---
        # map each timestep's multichannel sample to embedding
        # temporal_embed works on last dim: (B, T, C) -> (B, T, D)
        temporal = self.temporal_embed(x)                    # (B, T, D)
        temporal = self.temporal_pos_encoder(temporal)       # (B, T, D)
        temporal = self.temporal_transformer(temporal)       # (B, T, D)

        # --- Spatial stream ---
        x_spatial = rearrange(x, 'b t c -> b c t')           # (B, C, T)
        spatial = self.spatial_embed(x_spatial)              # (B, C, D)
        spatial = spatial + self.spatial_pos_encoder[:, :C, :]  # broadcast add (1, C, D)
        spatial = self.spatial_transformer(spatial)          # (B, C, D)

        # --- Fusion: temporal queries over spatial keys/values ---
        fused, _ = self.fusion_cross_attention(
            query=temporal,    # (B, T, D)
            key=spatial,       # (B, C, D)
            value=spatial      # (B, C, D)
        )  # fused: (B, T, D)
        fused = self.fusion_norm(fused + temporal)           # residual + layernorm -> (B, T, D)

        # --- Regression: per-timestep to pose_dim ---
        preds_norm = self.regression_head(fused)             # (B, T, pose_dim)

        if return_denorm:
            preds_denorm = self.denormalize(preds_norm)
            return preds_norm, preds_denorm
        return preds_norm
