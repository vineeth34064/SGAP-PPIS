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

        # Normalized adjacency with strict 1.0 diagonal
        adj_loop = adj.clone()
        adj_loop.fill_diagonal_(1.0)
        deg_inv_sqrt = torch.pow(adj_loop.sum(dim=-1).clamp(min=1e-6), -0.5)
        self.norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj_loop * deg_inv_sqrt.unsqueeze(0)

        self.targets = torch.randint(0, 2, (self.N,))

    def test_01_egnn_dimensions_and_summary(self):
        """Verify EGNN layer outputs, scale projections S^(k), and summary J."""
        egnn = EGNNBranch(in_dim=62, dg=256, da=128, num_layers=5)
        out = egnn(self.x, self.pos, self.edge_index, self.edge_attr)

        self.assertEqual(len(out['S_layers']), 5)
        for s_k in out['S_layers']:
            self.assertEqual(s_k.shape, (self.N, 128))
        self.assertEqual(out['J'].shape, (self.N, 128))
        self.assertEqual(out['final_h'].shape, (self.N, 256))
        self.assertEqual(out['final_pos'].shape, (self.N, 3))
        self.assertAlmostEqual(torch.sum(out['omega']).item(), 1.0, places=5)
        print("  [Pass] EGNN multi-scale geometric states and summary J verified.")

    def test_02_egnn_se3_equivariance(self):
        """Verify SE(3) equivariance: T(R p + t) = R T(p) + t and invariant node features."""
        layer = EGNNLayer(hidden_nf=64, edge_feat_dim=2, dropout=0.0)
        layer.eval()  # Disable dropout to verify pure geometric equivariance

        h = torch.randn(self.N, 64)
        p = self.pos.clone()

        # Generate random 3D orthogonal rotation matrix (det = +1)
        angle = np.pi / 4
        rot_z = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle),  np.cos(angle), 0.0],
            [0.0,            0.0,           1.0]
        ], dtype=torch.float32)
        trans = torch.tensor([2.5, -1.0, 3.2], dtype=torch.float32)

        # Transformed inputs
        p_trans = torch.matmul(p, rot_z.t()) + trans

        with torch.no_grad():
            h_orig, p_orig, _, _ = layer(h, p, self.edge_index, self.edge_attr)
            h_rot, p_rot, _, _ = layer(h, p_trans, self.edge_index, self.edge_attr)

        # Expected rotated coordinates
        p_orig_rotated = torch.matmul(p_orig, rot_z.t()) + trans

        # Coordinates should be equivariant
        coord_diff = torch.max(torch.abs(p_rot - p_orig_rotated)).item()
        self.assertLess(coord_diff, 1e-4)

        # Node features should be invariant
        feat_diff = torch.max(torch.abs(h_rot - h_orig)).item()
        self.assertLess(feat_diff, 1e-4)
        print(f"  [Pass] EGNN SE(3) equivariance strictly verified (Coord max diff: {coord_diff:.2e}, Feat max diff: {feat_diff:.2e}).")

    def test_03_adaptive_appnp(self):
        """Verify Adaptive APPNP dynamic teleportation coefficients and propagation."""
        appnp = AdaptiveAPPNPBranch(in_dim=62, da=128, num_appnp_layers=5, num_egnn_layers=5)
        dummy_S = [torch.randn(self.N, 128) for _ in range(5)]
        dummy_J = torch.randn(self.N, 128)

        out = appnp(self.x, self.norm_adj, dummy_S, dummy_J)
        self.assertEqual(out['H_out'].shape, (self.N, 5 * 128))
        self.assertEqual(len(out['all_alpha']), 5)

        # Verify alpha values lie strictly within [0.05, 0.95]
        for alpha in out['all_alpha']:
            self.assertTrue(torch.all(alpha >= 0.05))
            self.assertTrue(torch.all(alpha <= 0.95))
        print("  [Pass] Adaptive APPNP propagation and clamping verified.")

    def test_04_muse_module(self):
        """Verify MUSE attention, multi-scale conv, and pointwise FFN combination."""
        muse = MUSEModule(in_dim=768, hidden_dim=256, kernel_sizes=(1, 3, 5), use_spatial_patch=True)
        Z = torch.randn(self.N, 768)
        out_with_adj = muse(Z, norm_adj=self.norm_adj)
        self.assertEqual(out_with_adj.shape, (self.N, 256))

        out_no_adj = muse(Z, norm_adj=None)
        self.assertEqual(out_no_adj.shape, (self.N, 256))
        print("  [Pass] MUSE module dual-domain fusion verified.")

    def test_05_focal_loss(self):
        """Verify Focal Loss computation and surface penalty gradient consistency."""
        focal = FocalLoss(alpha_pos=0.75, alpha_neg=0.25, gamma=2.0)
        logits = torch.randn(self.N, 2)
        loss = focal(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Verify that true binding sites (y=1) are not penalized even if buried
        rsa = torch.full((self.N,), 0.01) # buried
        targets_pos = torch.ones(self.N)
        loss_unpen = focal(logits, targets_pos, rsa=None, surface_penalty_weight=0.0)
        loss_pen = focal(logits, targets_pos, rsa=rsa, surface_penalty_weight=1.0)
        self.assertAlmostEqual(loss_unpen.item(), loss_pen.item(), places=5)
        print(f"  [Pass] Focal Loss and surface penalty consistency verified: {loss.item():.4f}")

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
        """Verify metric calculations and temperature scaling on both 1D and (N, 2) logits."""
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.4, 0.3, 0.7, 0.15])
        metrics = compute_all_metrics(y_true, y_prob)
        self.assertIn('AUPRC', metrics)
        self.assertIn('MCC', metrics)
        self.assertIn('AUROC', metrics)

        best_th, best_mcc = find_optimal_threshold(y_true, y_prob, metric='MCC')
        self.assertGreaterEqual(best_mcc, metrics['MCC'])

        # 1D logits calibration
        scaler_1d = TemperatureScaling()
        val_logits_1d = torch.tensor([[-1.0], [2.0], [-0.5], [1.5]])
        val_targets = torch.tensor([0, 1, 0, 1])
        temp_1d = scaler_1d.fit(val_logits_1d, val_targets, max_iter=10)
        self.assertTrue(temp_1d > 0)

        # (N, 2) multi-class logits calibration (addresses EGM-02 blocker)
        scaler_2class = TemperatureScaling()
        val_logits_2class = torch.tensor([[-1.0, 1.0], [2.0, -1.0], [-0.5, 0.5], [1.5, -0.5]])
        temp_2class = scaler_2class.fit(val_logits_2class, val_targets, max_iter=10)
        self.assertTrue(temp_2class > 0)
        calibrated_probs = scaler_2class.calibrate_probs(val_logits_2class)
        self.assertEqual(calibrated_probs.shape[0], 4)
        print(f"  [Pass] Metrics, optimal thresholding, and temperature scaling verified (T_1d={temp_1d:.3f}, T_2class={temp_2class:.3f}).")

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
