"""
Adapter modules for parameter-efficient fine-tuning.
Includes FiLM adapters (feature-wise modulation) and Residual Action adapters (action-space correction).
"""

import torch
import torch.nn as nn


class FiLMAdapter(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) adapter that wraps a frozen linear layer.
    
    Applies affine transformation to the output of a frozen base layer:
        y = base(x) * (1 + α * γ) + α * β
    
    where γ (gamma) and β (beta) are predicted from context embedding e_t.
    
    Args:
        base_linear: Frozen linear layer to wrap
        ctx_dim: Dimension of context embedding from history encoder
        hidden: Hidden dimension for modulation network
        clamp_gamma: Optional clamping range for gamma values
        use_gate: Whether to use learnable gating parameter α
    
    The adapter is initialized to identity transformation (zero-init) for stable training.
    """
    def __init__(self, base_linear: nn.Linear, ctx_dim: int, hidden: int = 64,
                 clamp_gamma: float | None = 2.0, use_gate: bool = True):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)  # Freeze base layer

        d = base_linear.out_features
        # Modulation network: predicts gamma and beta from context
        self.mod = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.LayerNorm(hidden),  # Add normalization for stability
            nn.ReLU(),
            nn.Linear(hidden, 2 * d)
        )
        # Zero initialization ensures identity at start
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

        self.clamp_gamma = clamp_gamma
        self.use_gate = use_gate
        if use_gate:
            # Learnable gate α: starts at 0.1
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, e):
        """
        Forward pass with context modulation.
        
        Args:
            x: Input tensor [batch_size, in_features]
            e: Context embedding [batch_size, ctx_dim]
        
        Returns:
            Modulated output [batch_size, out_features]
        """
        y = self.base(x)  # Pre-activation from frozen base
        gamma, beta = self.mod(e).chunk(2, dim=-1)

        if self.clamp_gamma is not None:
            gamma = torch.clamp(gamma, -self.clamp_gamma, self.clamp_gamma)

        if self.use_gate:
            y = y * (1.0 + self.alpha * gamma) + self.alpha * beta
        else:
            y = y * (1.0 + gamma) + beta
        return y


class ResidualActionAdapter(nn.Module):
    """
    Residual Action Adapter that adds learned residual actions to a frozen base policy.

    Unlike FiLM which modulates intermediate features, this adapter operates directly in
    action space by predicting residual corrections:
        a_final = a_base + α * a_residual

    where a_base comes from the frozen base policy and a_residual is predicted from
    the concatenation of context embedding e_t and proprioceptive observations via a
    learnable MLP.

    Args:
        num_actions: Dimension of action space
        ctx_dim: Dimension of context embedding from history encoder
        proprio_dim: Dimension of proprioceptive observations (policy_obs used by base policy)
        hidden_dims: List of hidden layer dimensions for residual MLP
        use_gate: Whether to use learnable gating parameter α
        clamp_residual: Optional clamping range for residual actions

    The adapter is initialized to zero residuals for stable training (identity at start).
    """
    def __init__(self, num_actions: int, ctx_dim: int, proprio_dim: int = 0,
                 hidden_dims: list[int] = [128, 64],
                 use_gate: bool = True, clamp_residual: float | None = None):
        super().__init__()
        self.num_actions = num_actions
        self.clamp_residual = clamp_residual
        self.use_gate = use_gate
        self.proprio_dim = proprio_dim

        # Build residual MLP: [context_embedding, proprio] -> residual action
        mlp_input_dim = ctx_dim + proprio_dim
        layers = []
        in_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))  # Add normalization for stability
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))

        self.residual_mlp = nn.Sequential(*layers)

        # Zero initialization ensures identity at start (no residual)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        if use_gate:
            # Learnable gate α: starts at 0.1 for gradual adaptation
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, base_action, e, proprio=None):
        """
        Add residual action to base action.

        Args:
            base_action: Base policy actions [batch_size, num_actions]
            e: Context embedding [batch_size, ctx_dim]
            proprio: Proprioceptive observations [batch_size, proprio_dim] (optional)

        Returns:
            Final action [batch_size, num_actions]
        """
        # Concatenate context embedding with proprio observations
        if proprio is not None and self.proprio_dim > 0:
            mlp_input = torch.cat([e, proprio], dim=-1)
        else:
            mlp_input = e

        residual = self.residual_mlp(mlp_input)

        if self.clamp_residual is not None:
            residual = torch.clamp(residual, -self.clamp_residual, self.clamp_residual)

        if self.use_gate:
            return base_action + self.alpha * residual
        else:
            return base_action + residual
