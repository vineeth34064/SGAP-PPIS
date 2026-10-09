import torch
import torch.nn as nn
import torch.nn.functional as F

class RadialBasisFunction(nn.Module):
    """
    Gaussian Radial Basis Function (RBF) expansion for inter-residue distances.
    Rectifies Gap 5 (Edge feature underutilization).
    """
    def __init__(self, num_rbf=16, cutoff=14.0):
        super(RadialBasisFunction, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        # Centers evenly spaced between 0 and cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / ((cutoff / num_rbf) ** 2)

    def forward(self, dist):
        """
        Args:
            dist: (M, 1) or (M,) Euclidean distances
        Returns:
            (M, num_rbf) Gaussian RBF features
        """
        dist = dist.view(-1, 1)
        return torch.exp(-self.gamma * (dist - self.centers) ** 2)


class EGNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) Layer with Edge Attention Gating
    as defined in Equations (3) - (7) of SGAP-PPIS paper:
      - Edge message: m_ij = phi_e(h_i, h_j, ||p_i - p_j||^2, e_ij)
      - Edge attention: m_tilde_ij = sigma(phi_att(m_ij)) * m_ij
      - Coordinate update: p_i^(k+1) = p_i^(k) + 1/|N(i)| sum (p_i - p_j) * phi_p(m_tilde_ij)
      - Node update: h_i^(k+1) = phi_h([h_i, m_i]) + h_i
    """
    def __init__(
        self,
        hidden_nf=256,
        edge_feat_dim=2,
        act_fn=nn.ReLU(),
        dropout=0.1,
        use_rbf=False,
        num_rbf=16,
        coord_tanh=True
    ):
        super(EGNNLayer, self).__init__()
        self.hidden_nf = hidden_nf
        self.use_rbf = use_rbf
        self.coord_tanh = coord_tanh

        if use_rbf:
            self.rbf = RadialBasisFunction(num_rbf=num_rbf, cutoff=14.0)
            dist_dim = num_rbf
        else:
            self.rbf = None
            dist_dim = 1  # squared distance ||p_i - p_j||^2

        edge_in_dim = 2 * hidden_nf + dist_dim + edge_feat_dim

        # phi_e: Edge message MLP (Eq. 3)
        self.phi_e = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_nf),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # phi_att: Edge-wise attention gate (Eq. 4)
        self.phi_att = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid()
        )

        # phi_p: Coordinate update MLP (Eq. 5)
        coord_layers = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        ]
        if coord_tanh:
            coord_layers.append(nn.Tanh())
        self.phi_p = nn.Sequential(*coord_layers)

        # Initialize coordinate update weights with small gain for stability
        with torch.no_grad():
            self.phi_p[2].weight.data.uniform_(-0.001, 0.001)

        # phi_h: Node feature update MLP (Eq. 7)
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf)
        )

        self.layer_norm = nn.LayerNorm(hidden_nf)

    def forward(self, h, p, edge_index, edge_attr=None):
        """
        Args:
            h: (N, hidden_nf) residue representations
            p: (N, 3) pseudo-coordinates
            edge_index: (2, M) [src, dst]
            edge_attr: (M, edge_feat_dim) edge features
        Returns:
            h_next: (N, hidden_nf)
            p_next: (N, 3)
            m_tilde: (M, hidden_nf) attention-gated edge messages
            att_weights: (M, 1) edge attention scores
        """
        src, dst = edge_index[0], edge_index[1]
        N = h.size(0)
        M = edge_index.size(1)

        # Coordinate difference & distance
        p_diff = p[src] - p[dst]  # (M, 3)
        dist_sq = torch.sum(p_diff ** 2, dim=-1, keepdim=True)  # (M, 1)

        if self.use_rbf:
            dist_feat = self.rbf(torch.sqrt(dist_sq.clamp(min=1e-8)))
        else:
            dist_feat = dist_sq / 196.0  # Normalized by 14^2 = 196 for scale stability

        # Edge feature concatenation
        if edge_attr is not None:
            edge_in = torch.cat([h[src], h[dst], dist_feat, edge_attr], dim=-1)
        else:
            edge_in = torch.cat([h[src], h[dst], dist_feat], dim=-1)

        # Eq. 3: Edge message m_ij
        m_ij = self.phi_e(edge_in)

        # Eq. 4: Edge-wise message gating m_tilde_ij = sigma(phi_att(m_ij)) * m_ij
        att_weights = self.phi_att(m_ij)
        m_tilde = att_weights * m_ij  # (M, hidden_nf)

        # Eq. 5: Coordinate update
        # p_i^(k+1) = p_i^(k) + 1/|N(i)| * sum_{j in N(i)} (p_i - p_j) * phi_p(m_tilde_ij)
        coord_weights = self.phi_p(m_tilde)  # (M, 1)
        coord_trans = p_diff * coord_weights  # (M, 3)

        # Mean aggregation over neighbors for coordinates
        p_agg = torch.zeros_like(p)
        deg = torch.zeros(N, 1, device=p.device, dtype=p.dtype)
        p_agg.scatter_add_(0, src.unsqueeze(-1).expand(-1, 3), coord_trans)
        deg.scatter_add_(0, src.unsqueeze(-1), torch.ones_like(coord_weights))
        p_update = p_agg / deg.clamp(min=1.0)
        p_next = p + p_update

        # Eq. 6: Sum aggregation of messages for nodes
        m_i = torch.zeros(N, self.hidden_nf, device=h.device, dtype=h.dtype)
        m_i.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.hidden_nf), m_tilde)

        # Eq. 7: Node feature update with residual connection
        # h_bar^(k+1) = phi_h([h_i, m_i]), h^(k+1) = h_bar^(k+1) + h_i
        node_in = torch.cat([h, m_i], dim=-1)
        h_bar = self.phi_h(node_in)
        h_next = self.layer_norm(h + h_bar)

        return h_next, p_next, m_tilde, att_weights


class EGNNBranch(nn.Module):
    """
    EGNN Geometric Encoding Branch of SGAP-PPIS (Section II-D).
    Extracts multi-scale geometric states S^(k) across K layers and computes
    the multi-scale geometric summary J = sum_{k=1}^K omega_k * S^(k).
    """
    def __init__(
        self,
        in_dim=62,
        dg=256,
        da=128,
        num_layers=5,
        edge_feat_dim=2,
        dropout=0.3,
        use_rbf=False
    ):
        super(EGNNBranch, self).__init__()
        self.in_dim = in_dim
        self.dg = dg
        self.da = da
        self.num_layers = num_layers

        # Eq. 2: Initial geometric projection H_g^(0) = ReLU(X W_g + b_g)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, dg),
            nn.ReLU()
        )

        # K EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_nf=dg,
                edge_feat_dim=edge_feat_dim,
                dropout=dropout,
                use_rbf=use_rbf
            )
            for _ in range(num_layers)
        ])

        # Layer-wise linear mapping S^(k) = ReLU(H_g^(k) W_s^(k) + b_s^(k)) in R^(N x da)
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dg, da),
                nn.ReLU()
            )
            for _ in range(num_layers)
        ])

        # Learnable layer aggregation weights omega_k for summary J (Eq. 8)
        self.omega = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Args:
            x: (N, in_dim=62) node feature matrix
            pos: (N, 3) residue pseudo-coordinates
            edge_index: (2, M)
            edge_attr: (M, 2)
        Returns:
            dict containing:
              - 'S_layers': list of K mapped states [S^(1), ..., S^(K)], each (N, da)
              - 'J': multi-scale geometric summary (N, da)
              - 'final_h': (N, dg)
              - 'final_pos': (N, 3)
              - 'all_pos': list of coordinates per layer
              - 'edge_att_layers': list of attention weights per layer (for dynamic coupling)
              - 'edge_msg_layers': list of gated edge messages per layer
        """
        h = self.input_proj(x)
        p = pos

        S_layers = []
        all_pos = [p]
        edge_att_layers = []
        edge_msg_layers = []

        for k, (layer, proj) in enumerate(zip(self.layers, self.scale_projections)):
            h, p, m_tilde, att = layer(h, p, edge_index, edge_attr)
            s_k = proj(h)
            S_layers.append(s_k)
            all_pos.append(p)
            edge_att_layers.append(att)
            edge_msg_layers.append(m_tilde)

        # Eq. 8: J = sum_{k=1}^K omega_k * S^(k)
        norm_omega = F.softmax(self.omega, dim=0)
        J = torch.zeros_like(S_layers[0])
        for k in range(self.num_layers):
            J = J + norm_omega[k] * S_layers[k]

        return {
            'S_layers': S_layers,
            'J': J,
            'final_h': h,
            'final_pos': p,
            'all_pos': all_pos,
            'edge_att_layers': edge_att_layers,
            'edge_msg_layers': edge_msg_layers,
            'omega': norm_omega
        }

