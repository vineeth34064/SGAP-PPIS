import os
import sys
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset

from data.dataset import PPISDataset
from models.sgap_ppis import SGAP_PPIS
from losses.focal_loss import FocalLoss
from utils.metrics import compute_all_metrics, find_optimal_threshold
from utils.calibration import TemperatureScaling

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_eval_model(
    model_mode='baseline',
    num_train_samples=8,
    num_val_samples=4,
    num_test_samples=10,
    epochs=3,
    lr=0.0001,
    seed=42,
    device='cpu'
):
    set_seed(seed)
    print(f"\n=======================================================")
    print(f"  Training & Evaluating Mode: {model_mode.upper()}")
    print(f"=======================================================")

    is_rectified = (model_mode == 'rectified')

    # Load datasets
    train_full = PPISDataset(split='Train_335-1')
    test60_full = PPISDataset(split='Test_60')
    test315_full = PPISDataset(split='Test_315-28')
    ubtest31_full = PPISDataset(split='UBtest_31-6')

    # Subsets for controlled smoke test
    train_ds = Subset(train_full, range(num_train_samples))
    val_ds = Subset(train_full, range(num_train_samples, num_train_samples + num_val_samples))
    test60_ds = Subset(test60_full, range(min(num_test_samples, len(test60_full))))
    test315_ds = Subset(test315_full, range(min(num_test_samples, len(test315_full))))
    ubtest31_ds = Subset(ubtest31_full, range(min(num_test_samples, len(ubtest31_full))))

    # Initialize model
    model = SGAP_PPIS(
        in_dim=62,
        dg=256,
        da=128,
        num_egnn_layers=5,
        num_appnp_layers=5,
        dropout=0.3,
        rectified=is_rectified
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha_pos=0.75, alpha_neg=0.25, gamma=2.0)

    best_val_auprc = -1.0
    best_weights = None

    print(f"Training for {epochs} epochs on {len(train_ds)} proteins...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for i in range(len(train_ds)):
            sample = train_ds[i]
            x = sample['x'].to(device)
            pos = sample['pos'].to(device)
            edge_index = sample['edge_index'].to(device)
            edge_attr = sample['edge_attr'].to(device)
            norm_adj = sample['norm_adj'].to(device)
            y = sample['y'].to(device)
            rsa = sample['rsa'].to(device) if 'rsa' in sample else None

            optimizer.zero_grad()
            out = model(x, pos, edge_index, edge_attr, norm_adj)
            surface_pen = 0.1 if is_rectified else 0.0
            loss = criterion(out['logits'], y, rsa=rsa, surface_penalty_weight=surface_pen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation evaluation
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                sample = val_ds[i]
                out = model(
                    sample['x'].to(device),
                    sample['pos'].to(device),
                    sample['edge_index'].to(device),
                    sample['edge_attr'].to(device),
                    sample['norm_adj'].to(device)
                )
                val_targets.extend(sample['y'].numpy().tolist())
                val_probs.extend(out['probs'][:, 1].cpu().numpy().tolist())

        val_metrics = compute_all_metrics(val_targets, val_probs)
        elapsed = time.time() - t0
        print(f"Epoch {ep:02d} [{elapsed:.1f}s] - Train Loss: {total_loss/len(train_ds):.4f} - Val AUPRC: {val_metrics['AUPRC']:.4f} - Val MCC: {val_metrics['MCC']:.4f}")

        if val_metrics['AUPRC'] > best_val_auprc:
            best_val_auprc = val_metrics['AUPRC']
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    model.load_state_dict(best_weights)
    model.eval()

    # Evaluate on all three independent test subsets
    results = {}
    for test_name, ds in [('Test_60', test60_ds), ('Test_315-28', test315_ds), ('UBtest_31-6', ubtest31_ds)]:
        targets, probs = [], []
        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds[i]
                out = model(
                    sample['x'].to(device),
                    sample['pos'].to(device),
                    sample['edge_index'].to(device),
                    sample['edge_attr'].to(device),
                    sample['norm_adj'].to(device)
                )
                targets.extend(sample['y'].numpy().tolist())
                probs.extend(out['probs'][:, 1].cpu().numpy().tolist())

        # If rectified, use optimal thresholding tuned on validation fold
        if is_rectified:
            best_th, _ = find_optimal_threshold(val_targets, val_probs, metric='MCC')
        else:
            best_th = 0.5  # Base paper fixed threshold

        m = compute_all_metrics(targets, probs, threshold=best_th)
        results[test_name] = m

    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(" SGAP-PPIS CONTROLLED SMOKE TEST COMPARISON: BASELINE vs RECTIFIED")
    print("=" * 70)

    # 1. Run Baseline (Original Paper Implementation)
    baseline_res = train_and_eval_model(
        model_mode='baseline',
        num_train_samples=8,
        num_val_samples=4,
        num_test_samples=10,
        epochs=3,
        seed=42,
        device=device
    )

    # 2. Run Rectified (SGAP-PPIS-Plus with all gap fixes)
    rectified_res = train_and_eval_model(
        model_mode='rectified',
        num_train_samples=8,
        num_val_samples=4,
        num_test_samples=10,
        epochs=3,
        seed=42,
        device=device
    )

    # Published Base Paper Results (Table III & Table IV)
    paper_results = {
        'Test_60': {'AUPRC': 0.670, 'MCC': 0.555, 'AUROC': 0.894, 'F1': 0.627, 'Precision': 0.612, 'Recall': 0.645},
        'Test_315-28': {'AUPRC': 0.625, 'MCC': 0.529},
        'UBtest_31-6': {'AUPRC': 0.408, 'MCC': 0.368}
    }

    print("\n" + "=" * 70)
    print("                   FINAL PERFORMANCE COMPARISON REPORT")
    print("=" * 70)

    for split in ['Test_60', 'Test_315-28', 'UBtest_31-6']:
        print(f"\n>>> Dataset: {split} <<<")
        print(f"{'Metric':<12} | {'Paper Base (Pub)':<18} | {'Baseline (Smoke)':<18} | {'Rectified (Smoke)':<18} | {'Delta vs Base':<14} | {'Status'}")
        print("-" * 105)

        base_split = baseline_res[split]
        rect_split = rectified_res[split]
        paper_split = paper_results.get(split, {})

        for metric in ['AUPRC', 'AUROC', 'MCC', 'F1', 'Precision', 'Recall']:
            if metric not in base_split:
                continue

            base_val = base_split[metric]
            rect_val = rect_split[metric]
            paper_val_str = f"{paper_split[metric]:.3f}" if metric in paper_split else "N/A"

            delta = rect_val - base_val
            pct = (delta / max(abs(base_val), 1e-6)) * 100

            if delta > 0.001:
                status = f" IMPROVED (+{pct:.1f}%)"
            elif delta < -0.001:
                status = f" DECREASED ({pct:.1f}%)"
            else:
                status = " EQUAL"

            delta_str = f"{delta:+.4f}"
            print(f"{metric:<12} | {paper_val_str:<18} | {base_val:<18.4f} | {rect_val:<18.4f} | {delta_str:<14} | {status}")

    # Save comparison to results
    os.makedirs('results', exist_ok=True)
    with open('results/smoke_test_comparison.json', 'w') as f:
        json.dump({
            'baseline': baseline_res,
            'rectified': rectified_res,
            'published_paper': paper_results
        }, f, indent=2)

    print("\n[+] Full comparison metrics written to 'results/smoke_test_comparison.json'.")

if __name__ == '__main__':
    main()

