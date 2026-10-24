import torch
import torch.nn as nn
import torch.nn.functional as F

class PLMEvolutionaryFusion(nn.Module):
    """
    Learned Gated Fusion for Pretrained Protein Language Model (pLM/ESM-2) Embeddings
    with Handcrafted Evolutionary Features (Gap 4 Rectification).

    Bridges the gap between classical bioinformatics features (PSSM+HMM)
    and modern contextualized self-supervised protein representations (ESM-2).
    """
    def __init__(self, evol_dim=40, plm_dim=1280, d_proj=128):
        super(PLMEvolutionaryFusion, self).__init__()
        self.evol_dim = evol_dim
        self.plm_dim = plm_dim
        self.d_proj = d_proj

        self.proj_evol = nn.Sequential(
            nn.Linear(evol_dim, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU()
        )

        self.proj_plm = nn.Sequential(
            nn.Linear(plm_dim, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU()
        )

        # Per-residue biophysical gate g_i in [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj // 2),
            nn.ReLU(),
            nn.Linear(d_proj // 2, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(d_proj, evol_dim)

    def forward(self, x_62, plm_feats=None):
        """
        Args:
            x_62: (N, 62) where first 40 dims are PSSM (20) + HMM (20)
                  and remaining 22 dims are DSSP (14) + resAF (7) + PPE (1)
            plm_feats: (N, 1280) ESM-2 embeddings
        Returns:
            fused_x: (N, 62) feature matrix enriched with pLM semantics
            gate_val: (N, 1) gate values
        """
        if plm_feats is None:
            return x_62, None

        evol_feats = x_62[:, :self.evol_dim]   # (N, 40)
        struct_feats = x_62[:, self.evol_dim:] # (N, 22)

        h_evol = self.proj_evol(evol_feats)    # (N, d_proj)
        h_plm = self.proj_plm(plm_feats)       # (N, d_proj)

        gate = self.gate_mlp(torch.cat([h_evol, h_plm], dim=-1)) # (N, 1)

        fused_evol = gate * h_evol + (1.0 - gate) * h_plm        # (N, d_proj)
        fused_evol_out = self.out_proj(fused_evol)               # (N, 40)

        # Preserve structural streams untouched, enriching evolutionary stream
        fused_x = torch.cat([fused_evol_out, struct_feats], dim=-1) # (N, 62)

        return fused_x, gate

