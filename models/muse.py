import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """
    1D Depthwise Separable Convolution:
    Depthwise convolution across residue length followed by pointwise (1x1) projection.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv1d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (1, in_channels, N) or (B, C, N)
        """
        out = self.depthwise(x)
        out = self.act(out)
        out = self.pointwise(out)
        return out


class SpatialPatchConv(nn.Module):
    """
    3D Spatial Surface Patch Convolution (Gap 3 Rectification).
    Captures conformational (discontinuous) interaction patches in 3D space
    alongside 1D sequence motifs.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialPatchConv, self).__init__()
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_nbr = nn.Linear(in_channels, out_channels)
        self.act = nn.GELU()

    def forward(self, x, norm_adj):
        """
        x: (N, in_channels)
        norm_adj: (N, N) normalized spatial contact matrix
        """
        h_self = self.linear_self(x)
        h_nbr = self.linear_nbr(torch.matmul(norm_adj, x))
        return self.act(h_self + h_nbr)


class MUSEModule(nn.Module):
    """
    MUSE (Multi-scale Attention Network) Module (Section II-F).
    Fuses topological diffusion information and geometric context by summing three parallel branches:
      1. Self-Attention Branch (Eq. 18): captures global long-range residue dependencies.
      2. Multi-Scale Convolution Branch (Eq. 19-20): depthwise separable convolutions with
         kernel sizes {1, 3, 5} along the residue dimension, dynamically weighted.
         [In Rectified mode: augmented with 3D Spatial Surface Patch Convolution].
      3. Position-wise Feed-Forward Network (Eq. 21): residue-level transformation.

    Fused output (Eq. 22):
      Zmuse = Attention(Z) + Conv(Z) + Pointwise(Z)
    """
    def __init__(
        self,
        in_dim=768,      # (T + 1) * da = 6 * 128 = 768
        hidden_dim=256,
        num_heads=1,
        kernel_sizes=(1, 3, 5),
        dropout=0.3,
        use_spatial_patch=False  # Gap 3 Rectification
    ):
        super(MUSEModule, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_spatial_patch = use_spatial_patch

        # Linear projections for Value matrix used across attention and conv branches
        self.W_V = nn.Linear(in_dim, hidden_dim)

        # Branch 1: Self-Attention (Eq. 18)
        self.W_Q = nn.Linear(in_dim, hidden_dim)
        self.W_K = nn.Linear(in_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.attn_dropout = nn.Dropout(dropout)

        # Branch 2: Multi-Scale 1D Convolutions (Eq. 19-20)
        self.conv_layers = nn.ModuleList([
            DepthwiseSeparableConv1d(hidden_dim, hidden_dim, k)
            for k in kernel_sizes
        ])
        self.conv_out_proj = nn.Linear(hidden_dim, hidden_dim)

        num_kernels = len(kernel_sizes)
        if use_spatial_patch:
            # Add spatial 3D conformational patch convolution
            self.spatial_conv = SpatialPatchConv(hidden_dim, hidden_dim)
            num_kernels += 1
        else:
            self.spatial_conv = None

        # Learnable convolution weights beta_i (Eq. 20)
        self.beta = nn.Parameter(torch.zeros(num_kernels))

        # Branch 3: Position-wise Feed-Forward Network (Eq. 21)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Z, norm_adj=None):
        """
        Args:
            Z: (N, in_dim=768) concatenated representation [H_out || J]
            norm_adj: (N, N) spatial contact adjacency (used if use_spatial_patch=True)
        Returns:
            Z_muse: (N, hidden_dim) fused representation
        """
        N = Z.size(0)

        # Shared Value projection: Z W_V
        V = self.W_V(Z)  # (N, hidden_dim)

        # -------------------------------------------------------------
        # Branch 1: Self-Attention (Eq. 18)
        # -------------------------------------------------------------
        Q = self.W_Q(Z)  # (N, hidden_dim)
        K = self.W_K(Z)  # (N, hidden_dim)

        attn_scores = torch.matmul(Q, K.t()) * self.scale  # (N, N)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_out = self.W_O(torch.matmul(attn_probs, V))  # (N, hidden_dim)

        # -------------------------------------------------------------
        # Branch 2: Multi-Scale Convolutions (Eq. 19-20)
        # -------------------------------------------------------------
        # Input to 1D conv needs shape: (1, hidden_dim, N)
        V_conv_in = V.t().unsqueeze(0)  # (1, hidden_dim, N)

        conv_outputs = []
        for conv in self.conv_layers:
            c_out = conv(V_conv_in)  # (1, hidden_dim, N)
            c_out = c_out.squeeze(0).t()  # (N, hidden_dim)
            conv_outputs.append(c_out)

        # Gap 3: Spatial Patch Convolution for conformational discontinuous epitopes
        if self.use_spatial_patch and self.spatial_conv is not None and norm_adj is not None:
            spatial_out = self.spatial_conv(V, norm_adj)  # (N, hidden_dim)
            conv_outputs.append(spatial_out)

        # Dynamically weighted combination using normalized beta weights (Eq. 20)
        beta_weights = F.softmax(self.beta, dim=0)
        weighted_conv = torch.zeros_like(conv_outputs[0])
        for i, c_out in enumerate(conv_outputs):
            weighted_conv = weighted_conv + beta_weights[i] * c_out

        conv_out = self.conv_out_proj(weighted_conv)

        # -------------------------------------------------------------
        # Branch 3: Position-wise FFN (Eq. 21)
        # -------------------------------------------------------------
        pointwise_out = self.ffn(Z)  # (N, hidden_dim)

        # -------------------------------------------------------------
        # Summation of Three Branches (Eq. 22)
        # -------------------------------------------------------------
        Z_muse = attn_out + conv_out + pointwise_out
        Z_muse = self.norm(Z_muse)

        return Z_muse

