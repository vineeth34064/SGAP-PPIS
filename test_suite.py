import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import PPISDataset
from models.egnn import EGNNBranch, EGNNLayer, RadialBasisFunction
from models.appnp import AdaptiveAPPNPBranch
from models.muse import MUSEModule, DepthwiseSeparableConv1d, SpatialPatchConv
from models.plm_fusion import PLMEvolutionaryFusion
from models.sgap_ppis import SGAP_PPIS
from losses.focal_loss import FocalLoss
from utils.metrics import compute_all_metrics, find_optimal_threshold
from utils.calibration import TemperatureScaling

class TestSGAP_PPIS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.N = 30  # 30 residues
        self.x = torch.randn(self.N, 62)
        self.pos = torch.randn(self.N, 3)
        self.pos = self.pos - self.pos.mean(dim=0, keepdim=True)

        # Create dummy connected graph
        dist = torch.cdist(self.pos, self.pos)
        adj = (dist < 14.0).float()
        src, dst = torch.where(adj == 1.0)
        self.edge_index = torch.stack([src, dst], dim=0)
        M = self.edge_index.size(1)
        self.edge_attr = torch.rand(M, 2)

        # Normalized adjacency
        adj_loop = adj + torch.eye(self.N)
        deg_inv_sqrt = torch.pow(adj_loop.sum(dim=-1).clamp(min=1e-6), -0.5)
        self.norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj_loop * deg_inv_sqrt.unsqueeze(0)

        self.targets = torch.randint(0, 2, (self.N,))

    def test_01_egnn_dimensions_and_summary(self):
        """Verify EGNN layer outputs, scale projections S^(k), and summary J."""
        egnn = EGNNBranch(in_dim=62, dg=256, da=128, num_layers=5)
        out = egnn(self.x, self.pos, self.edge_index, self.edge_attr)

        self.assertEqual(len(out['S_layers']), 5)
        for s in out['S_layers']:
            self.assertEqual(s.shape, (self.N, 128))
        self.assertEqual(out['J'].shape, (self.N, 128))
        self.assertEqual(out['final_h'].shape, (self.N, 256))
        self.assertEqual(out['final_pos'].shape, (self.N, 3))
        print("  [Pass] EGNN dimensions and multi-scale summary verified.")

    def test_02_egnn_equivariance(self):
        """
        Verify SE(3) equivariance of coordinate updates and invariance of node representations.
        Applying rotation R and translation t:
        p' = p R^T + t
        h'(p') == h(p)
        p_next(p') == p_next(p) R^T + t
        """
        layer = EGNNLayer(hidden_nf=64, edge_feat_dim=2, coord_tanh=True)
        layer.eval()
        h = torch.randn(self.N, 64)
        p = self.pos.clone()

        # Generate random 3D rotation matrix via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        R = Q
        t = torch.tensor([[2.5, -1.0, 3.2]])

        p_trans = torch.matmul(p, R.t()) + t

        # Forward on original
        h_orig, p_orig, _, _ = layer(h, p, self.edge_index, self.edge_attr)

        # Forward on rotated and translated coordinates
        h_rot, p_rot, _, _ = layer(h, p_trans, self.edge_index, self.edge_attr)

        # Invariance of node features: h_rot should equal h_orig
        h_diff = torch.max(torch.abs(h_orig - h_rot)).item()
        self.assertLess(h_diff, 1e-4, f"Node features are not SE(3) invariant! Diff: {h_diff}")

        # Equivariance of coordinates: p_rot should equal p_orig * R^T + t
        p_expected = torch.matmul(p_orig, R.t()) + t
        p_diff = torch.max(torch.abs(p_rot - p_expected)).item()
        self.assertLess(p_diff, 1e-4, f"Coordinate update is not SE(3) equivariant! Diff: {p_diff}")
        print(f"  [Pass] EGNN SE(3) Equivariance verified (h diff: {h_diff:.2e}, p diff: {p_diff:.2e}).")

    def test_03_appnp_adaptive_propagation(self):
        """Verify adaptive APPNP branch multi-step propagation and scale alignment."""
        appnp = AdaptiveAPPNPBranch(in_dim=62, da=128, num_appnp_layers=5, num_egnn_layers=5)
        dummy_S = [torch.randn(self.N, 128) for _ in range(5)]
        dummy_J = torch.randn(self.N, 128)

        out = appnp(self.x, self.norm_adj, dummy_S, dummy_J)
        # H_out should concatenate all 5 steps: 5 * 128 = 640
        self.assertEqual(out['H_out'].shape, (self.N, 640))
        self.assertEqual(len(out['all_alpha']), 5)
        # Verify alpha values lie strictly within [0.05, 0.95]
        for alpha in out['all_alpha']:
            self.assertTrue(torch.all(alpha >= 0.05))
            self.assertTrue(torch.all(alpha <= 0.95))
        print("  [Pass] Adaptive APPNP propagation and clamping verified.")

    def test_04_muse_module(self):
        """Verify MUSE attention, multi-scale conv, and pointwise FFN combination."""
        muse = MUSEModule(in_dim=768, hidden_dim=256, kernel_sizes=(1, 3, 5))
        Z = torch.randn(self.N, 768)
        out = muse(Z)
        self.assertEqual(out.shape, (self.N, 256))
        print("  [Pass] MUSE module fusion verified.")

    def test_05_focal_loss(self):
        """Verify Focal Loss computation on imbalanced residue batch."""
        focal = FocalLoss(alpha_pos=0.75, alpha_neg=0.25, gamma=2.0)
        logits = torch.randn(self.N, 2)
        loss = focal(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        print(f"  [Pass] Focal Loss computation verified: {loss.item():.4f}")

    def test_06_ablation_modes(self):
        """Verify all ablation variants can instantiate and run cleanly."""
        modes = ['none', 'wo_scale_align', 'wo_multi_step_concat', 'wo_appnp', 'wo_egnn', 'wo_muse']
        for mode in modes:
            model = SGAP_PPIS(ablation=mode)
            out = model(self.x, self.pos, self.edge_index, self.edge_attr, self.norm_adj)
            self.assertEqual(out['logits'].shape, (self.N, 2))
            print(f"  [Pass] Ablation '{mode}' verified.")

    def test_07_rectified_mode(self):
        """Verify rectified mode with all critical gap fixes."""
        model = SGAP_PPIS(rectified=True, use_plm=True)
        plm_dummy = torch.randn(self.N, 1280)
        out = model(self.x, self.pos, self.edge_index, self.edge_attr, self.norm_adj, plm_feats=plm_dummy)
        self.assertEqual(out['logits'].shape, (self.N, 2))
        print("  [Pass] Rectified SGAP-PPIS-Plus model mode verified.")

    def test_08_metrics_and_calibration(self):
        """Verify metric calculations and temperature scaling."""
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.3, 0.7, 0.15])
        metrics = compute_all_metrics(y_true, y_prob)
        self.assertIn('AUPRC', metrics)
        self.assertIn('MCC', metrics)
        self.assertIn('AUROC', metrics)

        best_th, best_mcc = find_optimal_threshold(y_true, y_prob, metric='MCC')
        self.assertGreaterEqual(best_mcc, metrics['MCC'])

        scaler = TemperatureScaling()
        val_logits = torch.tensor([[-1.0], [2.0], [-0.5], [1.5]])
        val_targets = torch.tensor([0, 1, 0, 1])
        temp = scaler.fit(val_logits, val_targets, max_iter=10)
        self.assertTrue(temp > 0)
        print(f"  [Pass] Metrics, optimal thresholding, and temperature scaling verified (T={temp:.3f}).")

    def test_09_dataset_integrity(self):
        """Verify dataset loading and protein counts for Test_60 and Train_335-1."""
        ds_test60 = PPISDataset(split='Test_60')
        self.assertEqual(len(ds_test60), 60)
        sample = ds_test60[0]
        self.assertEqual(sample['x'].shape[1], 62)
        self.assertEqual(sample['pos'].shape[1], 3)
        self.assertEqual(sample['edge_attr'].shape[1], 2)
        print("  [Pass] Dataset integrity verified (Test_60 len: 60, feat dim: 62).")

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSGAP_PPIS)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
