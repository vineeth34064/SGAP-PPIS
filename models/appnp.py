import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAPPNPBranch(nn.Module):
    """
    Geometry-Conditioned Adaptive APPNP Branch (Section II-E).
    Performs multi-step approximate personalized propagation where propagation
    coefficients are dynamically generated from scale-aligned geometric states of the EGNN branch.

    Includes full ablation options:
      - scale_align: True (default) vs False (w/o ScaleAlign: uses J uniformly)
      - multi_step_concat: True (default) vs False (w/o Multi-Step Concat: uses only final H_a^(T))
      - fixed_alpha: None (adaptive) vs float in [0.0, 1.0] (SGAP_Fixed_alpha)

    Includes critical gap rectifications:
      - anisotropic_alpha: True (Gap 1: channel-wise alpha in R^(N x da) rather than scalar R^(N x 1))
      - dynamic_coupling: True (Gap 2: dynamically modulate diffusion graph using EGNN layer attention)
    """
    def __init__(
        self,
        in_dim=62,
        da=128,
        num_appnp_layers=5,
        num_egnn_layers=5,
        alpha_min=0.05,
        alpha_max=0.95,
        scale_align=True,
        multi_step_concat=True,
        fixed_alpha=None,
        anisotropic_alpha=False,
        dynamic_coupling=False
    ):
        super(AdaptiveAPPNPBranch, self).__init__()
        self.in_dim = in_dim
        self.da = da
        self.num_appnp_layers = num_appnp_layers
        self.num_egnn_layers = num_egnn_layers
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.scale_align = scale_align
        self.multi_step_concat = multi_step_concat
        self.fixed_alpha = fixed_alpha
        self.anisotropic_alpha = anisotropic_alpha
        self.dynamic_coupling = dynamic_coupling

        # Eq. 10: Root state projection H_a^(0) = ReLU(X W_0 + b_0)
        self.root_proj = nn.Sequential(
            nn.Linear(in_dim, da),
            nn.ReLU()
        )

        # Eq. 13: alpha_tilde^(t) = Sigmoid(W_alpha C^(t) + b_alpha)
        # If anisotropic_alpha=True (Gap 1), outputs da channels; otherwise scalar 1 channel
        alpha_out_dim = da if anisotropic_alpha else 1
        self.alpha_mlp = nn.Sequential(
            nn.Linear(da, da // 2),
            nn.ReLU(),
            nn.Linear(da // 2, alpha_out_dim),
            nn.Sigmoid()
        )

        # Dynamic Edge Modulation for Gap 2 Rectification
        if dynamic_coupling:
            self.edge_mod_mlp = nn.Sequential(
                nn.Linear(da, 1),
                nn.Sigmoid()
            )

    def _get_scale_index(self, t):
        """
        Eq. 11: idx(t) = 1 + round((t - 1) * (L_EGNN - 1) / (L_APPNP - 1))
        Zero-indexed: idx_zero = round(t * (L_EGNN - 1) / (L_APPNP - 1)) for t in [0, L_APPNP - 1]
        """
        if self.num_appnp_layers == 1 or self.num_egnn_layers == 1:
            return 0
        ratio = (self.num_egnn_layers - 1) / (self.num_appnp_layers - 1)
        idx = int(round(t * ratio))
        return min(max(idx, 0), self.num_egnn_layers - 1)

    def forward(
        self,
        x,
        norm_adj,
        egnn_states,
        egnn_summary_J,
        edge_index=None,
        edge_att_layers=None
    ):
        """
        Args:
            x: (N, in_dim=62) node feature matrix
            norm_adj: (N, N) symmetrically normalized adjacency matrix
            egnn_states: list of S^(k) from EGNN, each (N, da)
            egnn_summary_J: (N, da) multi-scale geometric summary J
            edge_index: (2, M) optional for dynamic coupling
            edge_att_layers: optional list of edge attention weights per EGNN layer
        Returns:
            dict containing:
              - 'H_out': (N, T * da) concatenated multi-step representations
              - 'all_H': list of all propagation states [H_a^(0), ..., H_a^(T)]
              - 'all_alpha': list of propagation coefficients per step
        """
        N = x.size(0)

        # Eq. 10: Root representation H_a^(0)
        H_root = self.root_proj(x)
        H_current = H_root

        H_steps = []
        all_alpha = []

        for t in range(self.num_appnp_layers):
            # Scale alignment (Eq. 11 & Eq. 12)
            if self.scale_align:
                scale_idx = self._get_scale_index(t)
                S_aligned = egnn_states[scale_idx]
                C_t = 0.5 * (S_aligned + egnn_summary_J)
            else:
                # Ablation w/o ScaleAlign: uniform shared geometric summary
                C_t = egnn_summary_J

            # Propagation coefficient alpha^(t)
            if self.fixed_alpha is not None:
                # Ablation SGAP_Fixed_alpha
                alpha_t = torch.full((N, 1), float(self.fixed_alpha), device=x.device, dtype=x.dtype)
            else:
                # Eq. 13 & 14: Adaptive residue-wise propagation coefficient
                alpha_raw = self.alpha_mlp(C_t)  # (N, 1) or (N, da)
                alpha_t = torch.clamp(alpha_raw, min=self.alpha_min, max=self.alpha_max)

            all_alpha.append(alpha_t)

            # Adjacency matrix for step t
            # If dynamic coupling (Gap 2): modulate normalized adjacency with scale-aligned EGNN states
            if self.dynamic_coupling and edge_index is not None and edge_att_layers is not None:
                scale_idx = self._get_scale_index(t)
                att_t = edge_att_layers[scale_idx]  # (M, 1)
                # Form modulated adjacency
                adj_mod = torch.zeros(N, N, device=x.device, dtype=x.dtype)
                adj_mod[edge_index[0], edge_index[1]] = att_t.squeeze(-1)
                # Symmetrize and combine with base norm_adj
                adj_mod = 0.5 * (adj_mod + adj_mod.t())
                cur_adj = 0.7 * norm_adj + 0.3 * adj_mod
                # Re-normalize row sums
                deg_inv_sqrt = torch.pow(cur_adj.sum(dim=-1).clamp(min=1e-6), -0.5)
                cur_adj = deg_inv_sqrt.unsqueeze(-1) * cur_adj * deg_inv_sqrt.unsqueeze(0)
            else:
                cur_adj = norm_adj

            # Eq. 15: H_a^(t+1) = (1 - alpha^(t)) * (A_hat * H_a^(t)) + alpha^(t) * H_a^(0)
            diffused = torch.matmul(cur_adj, H_current)
            H_next = (1.0 - alpha_t) * diffused + alpha_t * H_root

            H_steps.append(H_next)
            H_current = H_next

        # Eq. 16: Multi-step propagation state representation
        if self.multi_step_concat:
            # Concatenate all T intermediate states along feature dimension: R^(N x (T * da))
            H_out = torch.cat(H_steps, dim=-1)
        else:
            # Ablation w/o Multi-Step Concat: uses only the final APPNP output H_a^(T)
            # Replicated T times to match the downstream expected dimension
            H_out = torch.cat([H_steps[-1]] * self.num_appnp_layers, dim=-1)

        return {
            'H_out': H_out,
            'all_H': [H_root] + H_steps,
            'all_alpha': all_alpha
        }

