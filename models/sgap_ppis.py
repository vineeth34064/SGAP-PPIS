import torch
import torch.nn as nn
import torch.nn.functional as F

from models.egnn import EGNNBranch
from models.appnp import AdaptiveAPPNPBranch
from models.muse import MUSEModule
from models.plm_fusion import PLMEvolutionaryFusion

class SGAP_PPIS(nn.Module):
    """
    Structure-Guided Adaptive Propagation for Protein-Protein Interaction Site Prediction
    (SGAP-PPIS) Architecture (Zhu et al., 2026).

    Coordinates:
      1. Dual-Branch Backbone:
         - EGNN Branch: multi-scale equivariant geometric encoding (dg=256, da=128)
         - APPNP Branch: geometry-conditioned adaptive topological diffusion (da=128, T=5)
      2. MUSE Module: multi-scale attention fusion (attention + dynamic conv + pointwise FFN)
      3. Classification Head: residue-level PPIS prediction

    Ablation modes supported:
      - 'none': Full SGAP-PPIS model
      - 'wo_scale_align': w/o ScaleAlign (uses shared global geometric summary J)
      - 'wo_multi_step_concat': w/o Multi-Step Concat (uses only final APPNP state)
      - 'wo_appnp': w/o APPNP (EGNN geometric summary only)
      - 'wo_egnn': w/o EGNN (standard APPNP without geometric guidance)
      - 'wo_muse': w/o MUSE (direct linear/MLP fusion without self-attention and multi-scale conv)

    Gap Rectifications supported via `rectified=True`:
      - Gap 1: Channel-wise anisotropic adaptive propagation
      - Gap 2: Dynamic geometry-topology coupling
      - Gap 3: Dual-domain conformational 3D patch + 1D sequence MUSE
      - Gap 4: ESM-2 language model evolutionary gating
      - Gap 5: Continuous Gaussian RBF inter-residue distance modeling
    """
    def __init__(
        self,
        in_dim=62,
        dg=256,
        da=128,
        num_egnn_layers=5,
        num_appnp_layers=5,
        dropout=0.3,
        num_classes=2,
        ablation='none',
        fixed_alpha=None,
        rectified=False,
        use_plm=False
    ):
        super(SGAP_PPIS, self).__init__()
        self.in_dim = in_dim
        self.dg = dg
        self.da = da
        self.num_egnn_layers = num_egnn_layers
        self.num_appnp_layers = num_appnp_layers
        self.ablation = ablation
        self.fixed_alpha = fixed_alpha
        self.rectified = rectified
        self.use_plm = use_plm

        # Gap 4: Optional Pretrained Language Model Evolutionary Fusion
        if rectified and use_plm:
            self.plm_fusion = PLMEvolutionaryFusion(evol_dim=40, plm_dim=1280, d_proj=da)
        else:
            self.plm_fusion = None

        # 1. EGNN Branch (Section II-D)
        self.use_egnn = (ablation != 'wo_egnn')
        if self.use_egnn:
            self.egnn = EGNNBranch(
                in_dim=in_dim,
                dg=dg,
                da=da,
                num_layers=num_egnn_layers,
                dropout=dropout,
                use_rbf=rectified  # Gap 5 Rectification
            )
        else:
            self.egnn = None

        # 2. Geometry-Conditioned Adaptive APPNP Branch (Section II-E)
        self.use_appnp = (ablation != 'wo_appnp')
        if self.use_appnp:
            scale_align = (ablation != 'wo_scale_align')
            multi_step_concat = (ablation != 'wo_multi_step_concat')

            self.appnp = AdaptiveAPPNPBranch(
                in_dim=in_dim,
                da=da,
                num_appnp_layers=num_appnp_layers,
                num_egnn_layers=num_egnn_layers,
                scale_align=scale_align,
                multi_step_concat=multi_step_concat,
                fixed_alpha=fixed_alpha,
                anisotropic_alpha=rectified,  # Gap 1 Rectification
                dynamic_coupling=rectified    # Gap 2 Rectification
            )
        else:
            self.appnp = None

        # Feature concatenation dimension for MUSE
        # Full model: Z = [H_out (T*da) || J (da)] -> dim = (T+1)*da = 6 * 128 = 768
        if self.use_appnp and self.use_egnn:
            z_dim = (num_appnp_layers + 1) * da
        elif self.use_appnp and not self.use_egnn:
            z_dim = num_appnp_layers * da
        elif not self.use_appnp and self.use_egnn:
            z_dim = da
        else:
            z_dim = in_dim

        # 3. MUSE Module (Section II-F)
        self.use_muse = (ablation != 'wo_muse')
        if self.use_muse:
            self.muse = MUSEModule(
                in_dim=z_dim,
                hidden_dim=dg,  # 256
                num_heads=1,
                kernel_sizes=(1, 3, 5),
                dropout=dropout,
                use_spatial_patch=rectified  # Gap 3 Rectification
            )
            head_in_dim = dg
        else:
            self.muse = None
            self.fallback_proj = nn.Sequential(
                nn.Linear(z_dim, dg),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            head_in_dim = dg

        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, head_in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim // 2, num_classes)
        )

    def forward(
        self,
        x,
        pos,
        edge_index,
        edge_attr,
        norm_adj,
        plm_feats=None
    ):
        """
        Args:
            x: (N, in_dim=62) node feature matrix
            pos: (N, 3) pseudo-coordinates
            edge_index: (2, M) residue connectivity graph
            edge_attr: (M, 2) edge features
            norm_adj: (N, N) symmetrically normalized adjacency
            plm_feats: (N, 1280) optional ESM-2 embeddings
        Returns:
            dict containing:
              - 'logits': (N, num_classes) residue class logits
              - 'probs': (N, num_classes) softmax interaction probabilities
              - 'alpha': propagation coefficients
              - 'J': geometric summary
              - 'H_out': topological features
              - 'Z_muse': fused representations
        """
        # Gap 4: Learned evolutionary fusion with pLM (if enabled)
        if self.plm_fusion is not None and plm_feats is not None:
            x, _ = self.plm_fusion(x, plm_feats)

        # 1. EGNN Branch Execution
        if self.use_egnn:
            egnn_out = self.egnn(x, pos, edge_index, edge_attr)
            S_layers = egnn_out['S_layers']
            J = egnn_out['J']
            edge_att_layers = egnn_out['edge_att_layers']
        else:
            S_layers = None
            J = None
            edge_att_layers = None

        # 2. Adaptive APPNP Branch Execution
        if self.use_appnp:
            if self.use_egnn:
                appnp_out = self.appnp(
                    x=x,
                    norm_adj=norm_adj,
                    egnn_states=S_layers,
                    egnn_summary_J=J,
                    edge_index=edge_index,
                    edge_att_layers=edge_att_layers
                )
            else:
                # w/o EGNN: use zero geometric guidance
                dummy_S = [torch.zeros(x.size(0), self.da, device=x.device)] * self.num_egnn_layers
                dummy_J = torch.zeros(x.size(0), self.da, device=x.device)
                appnp_out = self.appnp(
                    x=x,
                    norm_adj=norm_adj,
                    egnn_states=dummy_S,
                    egnn_summary_J=dummy_J
                )
            H_out = appnp_out['H_out']
            all_alpha = appnp_out['all_alpha']
        else:
            H_out = None
            all_alpha = None

        # 3. Cross-Branch Concatenation: Z = [H_out || J] (Eq. 17)
        if self.use_appnp and self.use_egnn:
            Z = torch.cat([H_out, J], dim=-1)
        elif self.use_appnp and not self.use_egnn:
            Z = H_out
        elif not self.use_appnp and self.use_egnn:
            Z = J
        else:
            Z = x

        # 4. MUSE Fusion
        if self.use_muse:
            Z_muse = self.muse(Z, norm_adj=norm_adj)
        else:
            Z_muse = self.fallback_proj(Z)

        # 5. Classification Logits & Probabilities
        logits = self.classifier(Z_muse)
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'alpha': all_alpha,
            'J': J,
            'H_out': H_out,
            'Z_muse': Z_muse
        }

