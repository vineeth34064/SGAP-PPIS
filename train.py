import os
import sys
import time
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="SGAP-PPIS Training and Cross-Validation Pipeline")
    parser.add_argument('--model_mode', type=str, default='baseline', choices=['baseline', 'rectified'],
                        help="Model mode: 'baseline' (faithful paper implementation) or 'rectified' (with gap fixes)")
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'wo_scale_align', 'wo_multi_step_concat', 'wo_appnp', 'wo_egnn', 'wo_muse'],
                        help="Ablation experiment switch as defined in Table V")
    parser.add_argument('--fixed_alpha', type=float, default=None,
                        help="Fixed propagation coefficient for APPNP ablation sweep (e.g. 0.1, 0.3, 0.5, 0.7)")
    parser.add_argument('--epochs', type=int, default=50, help="Maximum epochs per fold")
    parser.add_argument('--lr', type=float, default=0.0001, help="Initial learning rate (paper: 0.0001)")
    parser.add_argument('--patience', type=int, default=5, help="Patience for lr decay")
    parser.add_argument('--lr_decay', type=float, default=0.3, help="Decay factor (paper: reduced by 70% -> x 0.3)")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate (paper: 0.3)")
    parser.add_argument('--dg', type=int, default=256, help="EGNN hidden dimension (paper: 256)")
    parser.add_argument('--da', type=int, default=128, help="APPNP hidden dimension (paper: 128)")
    parser.add_argument('--num_egnn_layers', type=int, default=5, help="EGNN layers (paper: 5)")
    parser.add_argument('--num_appnp_layers', type=int, default=5, help="APPNP layers (paper: 5)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--folds', type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('--single_fold', type=int, default=None, help="Train only a single fold index (0 to 4)")
    parser.add_argument('--smoke_test', action='store_true', help="Rapid smoke test on subset of data")
    parser.add_argument('--use_plm', action='store_true', help="Use ESM-2 language model embeddings (Gap 4)")
    parser.add_argument('--surface_penalty', type=float, default=0.0, help="Surface feasibility loss weight (Gap 7)")
    parser.add_argument('--coord_noise', type=float, default=0.0, help="Coordinate perturbation noise std (Gap 8)")
    parser.add_argument('--data_dir', type=str, default=None, help="Path to Dataset directory")
    parser.add_argument('--feature_dir', type=str, default=None, help="Path to Feature directory")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Directory to save logs")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_dataset(model, dataset, device, temp_scaler=None):
    model.eval()
    all_targets = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample['x'].to(device)
            pos = sample['pos'].to(device)
            edge_index = sample['edge_index'].to(device)
            edge_attr = sample['edge_attr'].to(device)
            norm_adj = sample['norm_adj'].to(device)
            y = sample['y'].numpy()
            plm = sample['plm'].to(device) if sample.get('plm') is not None else None

            out = model(x, pos, edge_index, edge_attr, norm_adj, plm_feats=plm)
            logits = out['logits']

            if temp_scaler is not None:
                probs = temp_scaler.calibrate_probs(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_targets.extend(y.tolist())
            all_probs.extend(probs.tolist())
            all_logits.append(logits.cpu())

    all_logits_tensor = torch.cat(all_logits, dim=0) if all_logits else None
    return np.array(all_targets), np.array(all_probs), all_logits_tensor, {}

def train_epoch(model, dataset, optimizer, criterion, device, surface_penalty=0.0):
    model.train()
    total_loss = 0.0

    # Shuffle training indices
    indices = np.random.permutation(len(dataset))
    for idx in indices:
        sample = dataset[idx]
        x = sample['x'].to(device)
        pos = sample['pos'].to(device)
        edge_index = sample['edge_index'].to(device)
        edge_attr = sample['edge_attr'].to(device)
        norm_adj = sample['norm_adj'].to(device)
        targets = sample['y'].to(device)
        rsa = sample['rsa'].to(device) if sample.get('rsa') is not None else None
        plm = sample['plm'].to(device) if sample.get('plm') is not None else None

        optimizer.zero_grad()
        out = model(x, pos, edge_index, edge_attr, norm_adj, plm_feats=plm)
        logits = out['logits']

        loss = criterion(logits, targets, rsa=rsa, surface_penalty_weight=surface_penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)

def train_fold(
    fold_idx,
    train_subset,
    val_subset,
    args,
    device,
    exp_name
):
    print(f"\n=================== Training Fold {fold_idx + 1} ===================")
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    is_rectified = (args.model_mode == 'rectified')
    model = SGAP_PPIS(
        in_dim=62,
        dg=args.dg,
        da=args.da,
        num_egnn_layers=args.num_egnn_layers,
        num_appnp_layers=args.num_appnp_layers,
        dropout=args.dropout,
        ablation=args.ablation,
        fixed_alpha=args.fixed_alpha,
        rectified=is_rectified,
        use_plm=args.use_plm
    ).to(device)

    criterion = FocalLoss(alpha_pos=0.75, alpha_neg=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    current_lr = args.lr
    best_val_auprc = -1.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = os.path.join(args.save_dir, f"{exp_name}_fold{fold_idx + 1}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model,
            train_subset,
            optimizer,
            criterion,
            device,
            surface_penalty=args.surface_penalty if is_rectified else 0.0
        )

        val_targets, val_probs, val_logits, _ = evaluate_dataset(model, val_subset, device)
        val_metrics = compute_all_metrics(val_targets, val_probs)
        val_auprc = val_metrics['AUPRC']
        val_mcc = val_metrics['MCC']

        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} - Val AUPRC: {val_auprc:.4f} - Val MCC: {val_mcc:.4f} - LR: {current_lr:.2e}")

        # Checkpoint selection on highest validation AUPRC (Section III-B)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0

            # 1. Post-hoc temperature calibration fit on validation logits (Gap 6)
            temp_scaler = TemperatureScaling().to(device)
            try:
                fitted_temp = temp_scaler.fit(val_logits.to(device), torch.tensor(val_targets).to(device), max_iter=20)
            except Exception:
                fitted_temp = 1.0

            # 2. Derive calibrated validation probabilities
            val_probs_calibrated = temp_scaler.calibrate_probs(val_logits.to(device)).cpu().numpy()

            # 3. Find optimal decision threshold on CALIBRATED probabilities (Gap 6)
            opt_thresh, opt_mcc = find_optimal_threshold(val_targets, val_probs_calibrated, metric='MCC')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'optimal_threshold': opt_thresh,
                'optimal_mcc': opt_mcc,
                'temperature': fitted_temp,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  --> [*] Best validation AUPRC achieved ({best_val_auprc:.4f}, Opt Thresh: {opt_thresh:.2f}, Temp: {fitted_temp:.2f}). Checkpoint saved.")
        else:
            patience_counter += 1
            # Learning rate decay strategy: reduced by 70% if val AUPRC does not improve for 5 consecutive epochs
            if patience_counter >= args.patience:
                new_lr = max(current_lr * args.lr_decay, args.min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    print(f"  --> [!] Decaying learning rate to: {current_lr:.2e}")
                patience_counter = 0

    print(f"Fold {fold_idx + 1} Complete. Best Epoch: {best_epoch}, Best Val AUPRC: {best_val_auprc:.4f}")
    return best_epoch, best_val_auprc, checkpoint_path

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=================================================================")
    print(f" SGAP-PPIS Training Pipeline")
    print(f" Mode: {args.model_mode.upper()}, Ablation: {args.ablation}")
    print(f" Device: {device}, Folds: {args.folds}, Epochs/fold: {args.epochs}")
    print("=================================================================")

    exp_name = f"sgap_{args.model_mode}_{args.ablation}"
    if args.fixed_alpha is not None:
        exp_name += f"_alpha{args.fixed_alpha}"

    # Load Train_335-1 dataset
    train_dataset = PPISDataset(
        split='Train_335-1',
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        load_plm=args.use_plm,
        coord_noise_std=args.coord_noise if args.model_mode == 'rectified' else 0.0
    )
    print(f"Loaded Train_335-1 with {len(train_dataset)} proteins.")

    # Load Test Sets for final reporting
    test60_ds = PPISDataset(split='Test_60', data_dir=args.data_dir, feature_dir=args.feature_dir, load_plm=args.use_plm)
    test315_ds = PPISDataset(split='Test_315-28', data_dir=args.data_dir, feature_dir=args.feature_dir, load_plm=args.use_plm)
    ubtest31_ds = PPISDataset(split='UBtest_31-6', data_dir=args.data_dir, feature_dir=args.feature_dir, load_plm=args.use_plm)

    if args.smoke_test:
        print("[!] SMOKE TEST MODE: Truncating datasets to small subsets.")
        train_dataset = Subset(train_dataset, range(min(6, len(train_dataset))))
        test60_ds = Subset(test60_ds, range(min(4, len(test60_ds))))
        test315_ds = Subset(test315_ds, range(min(4, len(test315_ds))))
        ubtest31_ds = Subset(ubtest31_ds, range(min(4, len(ubtest31_ds))))
        args.epochs = min(args.epochs, 2)
        args.folds = 2

    # 5-Fold Cross Validation Setup
    N_total = len(train_dataset)
    indices = np.arange(N_total)
    np.random.shuffle(indices)
    fold_splits = np.array_split(indices, args.folds)

    best_epochs = []
    best_auprcs = []
    checkpoint_paths = []

    folds_to_run = [args.single_fold] if args.single_fold is not None else range(args.folds)

    for fold_idx in folds_to_run:
        val_idx = fold_splits[fold_idx]
        train_idx = np.concatenate([fold_splits[j] for j in range(args.folds) if j != fold_idx])

        train_sub = Subset(train_dataset, train_idx)
        val_sub = Subset(train_dataset, val_idx)

        best_ep, best_val, ckpt = train_fold(
            fold_idx, train_sub, val_sub, args, device, exp_name
        )
        best_epochs.append(best_ep)
        best_auprcs.append(best_val)
        checkpoint_paths.append(ckpt)

    print("\n=================== Cross-Validation Summary ===================")
    print(f"Average Best Epoch across folds: {np.mean(best_epochs):.1f}")
    print(f"Mean Validation AUPRC across folds: {np.mean(best_auprcs):.4f}")

    # Evaluate best fold checkpoints on Independent Test Sets
    print("\n=================== Independent Test Evaluation ===================")
    for test_name, test_ds in [('Test_60', test60_ds), ('Test_315-28', test315_ds), ('UBtest_31-6', ubtest31_ds)]:
        test_metrics_list = []
        for ckpt in checkpoint_paths:
            ckpt_data = torch.load(ckpt, map_location=device)
            eval_model = SGAP_PPIS(
                in_dim=62,
                dg=args.dg,
                da=args.da,
                num_egnn_layers=args.num_egnn_layers,
                num_appnp_layers=args.num_appnp_layers,
                dropout=args.dropout,
                ablation=args.ablation,
                fixed_alpha=args.fixed_alpha,
                rectified=(args.model_mode == 'rectified'),
                use_plm=args.use_plm
            ).to(device)
            eval_model.load_state_dict(ckpt_data['model_state_dict'])

            temp_scaler = None
            if args.model_mode == 'rectified' and 'temperature' in ckpt_data:
                temp_scaler = TemperatureScaling(init_temp=ckpt_data['temperature']).to(device)

            eval_threshold = ckpt_data.get('optimal_threshold', 0.5) if (args.model_mode == 'rectified') else 0.5

            targets, probs, _, _ = evaluate_dataset(eval_model, test_ds, device, temp_scaler=temp_scaler)
            metrics = compute_all_metrics(targets, probs, threshold=eval_threshold)
            test_metrics_list.append(metrics)

        # Average metrics across evaluated checkpoints
        avg_metrics = {}
        for key in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUROC', 'AUPRC']:
            vals = [m[key] for m in test_metrics_list]
            avg_metrics[key] = (np.mean(vals), np.std(vals))

        print(f"\n--- {test_name} Results (Mean ± Std over {len(checkpoint_paths)} models) ---")
        for key in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUROC', 'AUPRC']:
            m_mean, m_std = avg_metrics[key]
            print(f"  {key:10s}: {m_mean:.4f} ± {m_std:.4f}")

    print("\nExecution complete.")

if __name__ == '__main__':
    main()
