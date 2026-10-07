"""
History encoder using GRU or Transformer for temporal context embedding.
Encodes sequences of observations into context vectors.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    GRU-based history encoder for generating context embeddings.

    Processes a sliding window of observations to generate temporal context
    that conditions the FiLM adapters in the actor network.
    
    The GRU processes the entire window from a zero state on each forward pass,
    making this non-recurrent from the training perspective.
    
    Args:
        in_dim: Per-timestep input dimension (e.g., single observation dimension)
                NOT the flattened history dimension
                Example: if obs is [x, y, z] per timestep, in_dim=3
                         (even if policy input is 5-step flattened = 15 dims)
        ctx_dim: Context embedding dimension (output)
        num_layers: Number of GRU layers
    """
    def __init__(self, in_dim, ctx_dim=64, num_layers=1):
        super().__init__()
        self.embed = nn.Linear(in_dim, ctx_dim)
        self.layer_norm = nn.LayerNorm(ctx_dim)  # Add layer normalization for stability
        self.gru = nn.GRU(input_size=ctx_dim, hidden_size=ctx_dim, num_layers=num_layers)
        self.ctx_dim = ctx_dim
        self.in_dim = in_dim

    def forward(self, window):
        """
        Forward pass through the encoder.
        
        Args:
            window: History window [seq_len, batch_size, in_dim]
                    Example: [32, 4096, 3] for 32-step history, 4096 envs, 3-dim obs per step
        
        Returns:
            e_t: Context embedding for current step [batch_size, ctx_dim]
                 Example: [4096, 64] for 4096 envs, 64-dim context
        """
        # Embed the entire window
        z = torch.tanh(self.embed(window))  # [seq_len, batch_size, ctx_dim]
        z = self.layer_norm(z)  # Apply layer normalization for stability
                
        # Process window from zero state
        y, _ = self.gru(z)

        # Return last timestep output as context
        e_t = y[-1]  # [batch_size, ctx_dim]
        return e_t


class TransformerEncoder(nn.Module):
    """
    Transformer-based history encoder for generating context embeddings.
    
    Processes a sliding window of observations using self-attention to generate
    temporal context that conditions the FiLM adapters in the actor network.
    
    Compared to GRU:
    - Better at capturing long-range dependencies
    - Parallel processing of sequences (faster training)
    - More interpretable attention patterns
    - Slightly more parameters
    
    The Transformer processes the entire window from scratch on each forward pass,
    making this non-recurrent from the training perspective.
    
    Args:
        in_dim: Per-timestep input dimension (e.g., single observation dimension)
                NOT the flattened history dimension
                Example: if obs is [x, y, z] per timestep, in_dim=3
                         (even if policy input is 5-step flattened = 15 dims)
        ctx_dim: Context embedding dimension (output)
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads (must divide ctx_dim evenly)
        dropout: Dropout rate for attention and feedforward layers
                 Default 0.0 (no dropout) - sufficient for RL with large parallel envs
                 Use 0.1-0.2 only if: small num_envs (<100), offline RL, or overfitting
        max_seq_len: Maximum sequence length for positional encoding
    """
    def __init__(self, in_dim, ctx_dim=64, num_layers=2, num_heads=4, 
                 dropout=0.0, max_seq_len=256):
        super().__init__()
        
        assert ctx_dim % num_heads == 0, f"ctx_dim ({ctx_dim}) must be divisible by num_heads ({num_heads})"
        
        self.ctx_dim = ctx_dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input embedding: project observations to context dimension
        self.embed = nn.Linear(in_dim, ctx_dim)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, 1, ctx_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Layer normalization before transformer
        self.input_norm = nn.LayerNorm(ctx_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim,
            nhead=num_heads,
            dim_feedforward=ctx_dim * 4,  # Standard practice: 4x hidden dim
            dropout=dropout,
            activation='gelu',  # GELU activation (modern standard)
            batch_first=False,  # [seq_len, batch_size, ctx_dim]
            norm_first=True     # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final layer norm for output stability
        self.output_norm = nn.LayerNorm(ctx_dim)
        
        # Optional: Learnable pooling weights for weighted average of sequence
        # (alternative to just taking last token)
        self.use_weighted_pool = False
        if self.use_weighted_pool:
            self.pool_weights = nn.Linear(ctx_dim, 1)

    def forward(self, window):
        """
        Forward pass through the encoder.
        
        Args:
            window: History window [seq_len, batch_size, in_dim]
                    Example: [32, 4096, 3] for 32-step history, 4096 envs, 3-dim obs per step
        
        Returns:
            e_t: Context embedding for current step [batch_size, ctx_dim]
                 Example: [4096, 64] for 4096 envs, 64-dim context
        """
        seq_len, batch_size, _ = window.shape
        
        # Embed observations to context dimension
        z = self.embed(window)  # [seq_len, batch_size, ctx_dim]
        
        # Add positional encoding (broadcast across batch)
        pos_enc = self.pos_encoding[:seq_len, :, :]  # [seq_len, 1, ctx_dim]
        z = z + pos_enc
        
        # Apply input normalization
        z = self.input_norm(z)
        
        # Process through transformer
        # No mask needed - we want full self-attention across the history
        y = self.transformer(z)  # [seq_len, batch_size, ctx_dim]
        
        # Apply output normalization
        y = self.output_norm(y)
        
        # Extract context embedding
        if self.use_weighted_pool:
            # Weighted pooling: learn which timesteps are important
            weights = torch.softmax(self.pool_weights(y).squeeze(-1), dim=0)  # [seq_len, batch_size]
            e_t = torch.sum(y * weights.unsqueeze(-1), dim=0)  # [batch_size, ctx_dim]
        else:
            # Simple approach: use last timestep (most recent observation)
            e_t = y[-1]  # [batch_size, ctx_dim]
        
        return e_t


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer encoder with masked self-attention.
    
    Unlike the standard TransformerEncoder, this uses causal masking where each
    timestep can only attend to previous timesteps (similar to autoregressive models).
    This can be useful when you want to strictly enforce temporal causality.
    
    For most RL applications, the standard TransformerEncoder (full attention) is
    preferred since we're processing complete history windows offline.
    
    Args:
        in_dim: Per-timestep input dimension
        ctx_dim: Context embedding dimension (output)
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads (must divide ctx_dim evenly)
        dropout: Dropout rate (default 0.0 - typically not needed for RL)
        max_seq_len: Maximum sequence length
    """
    def __init__(self, in_dim, ctx_dim=64, num_layers=2, num_heads=4, 
                 dropout=0.0, max_seq_len=256):
        super().__init__()
        
        assert ctx_dim % num_heads == 0, f"ctx_dim ({ctx_dim}) must be divisible by num_heads ({num_heads})"
        
        self.ctx_dim = ctx_dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input embedding
        self.embed = nn.Linear(in_dim, ctx_dim)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, 1, ctx_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Layer normalization
        self.input_norm = nn.LayerNorm(ctx_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim,
            nhead=num_heads,
            dim_feedforward=ctx_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(ctx_dim)
        
        # Register causal mask buffer (will be created on first forward pass)
        self.register_buffer('causal_mask', None)

    def _generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask (upper triangular)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, window):
        """
        Forward pass through the causal encoder.
        
        Args:
            window: History window [seq_len, batch_size, in_dim]
        
        Returns:
            e_t: Context embedding for current step [batch_size, ctx_dim]
        """
        seq_len, batch_size, _ = window.shape
        
        # Embed observations
        z = self.embed(window)  # [seq_len, batch_size, ctx_dim]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :, :]
        z = z + pos_enc
        
        # Apply input normalization
        z = self.input_norm(z)
        
        # Generate or reuse causal mask
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = self._generate_causal_mask(seq_len, window.device)
        
        # Process through transformer with causal mask
        y = self.transformer(z, mask=self.causal_mask)  # [seq_len, batch_size, ctx_dim]
        
        # Apply output normalization
        y = self.output_norm(y)
        
        # Return last timestep (only attends to past)
        e_t = y[-1]  # [batch_size, ctx_dim]
        
        return e_t
