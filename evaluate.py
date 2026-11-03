import os
import argparse
import json
import numpy as np
import torch

from data.dataset import PPISDataset
from models.sgap_ppis import SGAP_PPIS
from utils.metrics import compute_all_metrics, find_optimal_threshold
from utils.calibration import TemperatureScaling

def parse_args():
    parser = argparse.ArgumentParser(description="SGAP-PPIS Evaluation and Benchmark Comparison")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory containing fold checkpoints")
    parser.add_argument('--model_mode', type=str, default='baseline', choices=['baseline', 'rectified'])
    parser.add_argument('--ablation', type=str, default='none')
    parser.add_argument('--fixed_alpha', type=float, default=None)
    parser.add_argument('--use_plm', action='store_true')
    parser.add_argument('--use_calibration', action='store_true', help="Apply temperature scaling calibration (Gap 6)")
    parser.add_argument('--optimize_threshold', action='store_true', help="Tune classification threshold on validation set")
    parser.add_argument('--case_study', type=str, default='4BVXB', help="Protein ID for case study (paper: 4BVX chain B)")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save evaluation results")
    return parser.parse_args()

def evaluate_model_on_dataset(model, dataset, device, threshold=0.5, temp_scaler=None):
    model.eval()
    all_targets = []
    all_probs = []
    protein_predictions = {}

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            pid = sample['pid']
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
                logits = temp_scaler(logits)

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_targets.extend(y.tolist())
            all_probs.extend(probs.tolist())
            protein_predictions[pid] = {
                'targets': y.tolist(),
                'probs': probs.tolist(),
                'preds': (probs >= threshold).astype(int).tolist()
            }

    metrics = compute_all_metrics(all_targets, all_probs, threshold=threshold)
    return metrics, protein_predictions, np.array(all_targets), np.array(all_probs)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"--- SGAP-PPIS Model Evaluation ---")
    print(f"Device: {device}")
    print(f"Mode: {args.model_mode}, Ablation: {args.ablation}")

    # Discover checkpoints
    checkpoints = []
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoints.append(args.checkpoint)
    elif os.path.exists(args.checkpoint_dir):
        for f in os.listdir(args.checkpoint_dir):
            if f.endswith('.pt') and f"sgap_{args.model_mode}_{args.ablation}" in f:
                checkpoints.append(os.path.join(args.checkpoint_dir, f))

    print(f"Found {len(checkpoints)} checkpoints for evaluation.")

    # Initialize model
    model = SGAP_PPIS(
        in_dim=62,
        dg=256,
        da=128,
        num_egnn_layers=5,
        num_appnp_layers=5,
        ablation=args.ablation,
        fixed_alpha=args.fixed_alpha,
        rectified=(args.model_mode == 'rectified'),
        use_plm=args.use_plm
    ).to(device)

    # Load test datasets
    test_datasets = {
        'Test_60': PPISDataset(split='Test_60', load_plm=args.use_plm),
        'Test_315-28': PPISDataset(split='Test_315-28', load_plm=args.use_plm),
        'UBtest_31-6': PPISDataset(split='UBtest_31-6', load_plm=args.use_plm),
    }

    results_table = {}

    for split_name, ds in test_datasets.items():
        print(f"\nEvaluating on {split_name} ({len(ds)} proteins)...")
        if len(checkpoints) > 0:
            fold_metrics = []
            case_study_stats = None

            for ckpt in checkpoints:
                ckpt_data = torch.load(ckpt, map_location=device)
                model.load_state_dict(ckpt_data['model_state_dict'])

                threshold = 0.5
                metrics, preds_dict, _, _ = evaluate_model_on_dataset(
                    model, ds, device, threshold=threshold
                )
                fold_metrics.append(metrics)

                # Check case study protein (4BVXB)
                if args.case_study in preds_dict:
                    cs_data = preds_dict[args.case_study]
                    cs_y = np.array(cs_data['targets'])
                    cs_prob = np.array(cs_data['probs'])
                    case_study_stats = compute_all_metrics(cs_y, cs_prob, threshold=0.5)

            # Aggregate over folds
            agg = {}
            for metric_key in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUROC', 'AUPRC']:
                vals = [m[metric_key] for m in fold_metrics]
                agg[metric_key] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            results_table[split_name] = agg

            print(f"Results for {split_name}:")
            for k, v in agg.items():
                print(f"  {k:10s}: {v}")

            if case_study_stats is not None:
                print(f"\n--- Case Study: {args.case_study} (Table VI format) ---")
                print(f"  TP: {case_study_stats['TP']}, TN: {case_study_stats['TN']}, FP: {case_study_stats['FP']}, FN: {case_study_stats['FN']}")
        else:
            print("No saved checkpoints found. Evaluating initialized model as baseline verification.")
            metrics, preds_dict, _, _ = evaluate_model_on_dataset(model, ds, device)
            results_table[split_name] = metrics
            print(f"Results for {split_name}: {metrics}")

    # Save summary
    out_file = os.path.join(args.output_dir, f"eval_{args.model_mode}_{args.ablation}.json")
    with open(out_file, 'w') as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved evaluation summary to: {out_file}")

if __name__ == '__main__':
    main()

