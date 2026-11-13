import os
import sys
import gzip
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PDBList

from models.sgap_ppis import SGAP_PPIS

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def get_centroid(residue):
    """Retrieve side-chain centroid (or CA fallback for Glycine)."""
    sidechain_coords = []
    for atom in residue:
        atom_name = atom.get_name().strip()
        if atom_name not in ['N', 'CA', 'C', 'O']:
            sidechain_coords.append(atom.get_coord())

    resname = residue.get_resname().strip()
    if resname == 'GLY' or len(sidechain_coords) == 0:
        if 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            atoms = list(residue.get_atoms())
            return atoms[0].get_coord() if atoms else np.zeros(3)
    else:
        return np.mean(sidechain_coords, axis=0)

def parse_pdb_chain(pdb_file, chain_id):
    """Parse structure, filter standard residues, extract centroids and sequence."""
    # Decompress if gzipped (.gz or .ent.gz)
    actual_path = pdb_file
    if pdb_file.endswith('.gz'):
        uncompressed_path = pdb_file[:-3]
        if not os.path.exists(uncompressed_path):
            with gzip.open(pdb_file, 'rb') as f_in, open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        actual_path = uncompressed_path

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', actual_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {pdb_file}. Available: {[c.id for c in model]}")

    chain = model[chain_id]
    residues = [r for r in chain if r.id[0] == ' ' and r.get_resname().strip() in AA_3TO1]
    if not residues:
        raise ValueError(f"No standard amino acid residues found in chain '{chain_id}'.")

    seq = "".join([AA_3TO1[r.get_resname().strip()] for r in residues])
    res_ids = [r.id[1] for r in residues]
    res_names = [r.get_resname().strip() for r in residues]
    pos = np.array([get_centroid(r) for r in residues], dtype=np.float32)

    return seq, res_ids, res_names, pos

def normalize_adjacency(adj):
    """
    Symmetric normalization of adjacency with self-loops:
    A_hat = D^(-1/2) (A + I_N) D^(-1/2)
    Enforces diagonal elements strictly equal 1.0 (no double-counting of self-loops).
    """
    N = adj.shape[0]
    adj_loop = np.copy(adj).astype(np.float32)
    np.fill_diagonal(adj_loop, 1.0)
    rowsum = adj_loop.sum(axis=1)
    r_inv = np.power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv @ adj_loop @ r_mat_inv

def align_len(arr, target_len):
    """Safely pad or truncate feature arrays to match sequence length."""
    if arr.shape[0] > target_len:
        return arr[:target_len]
    elif arr.shape[0] < target_len:
        pad = np.zeros((target_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        return np.vstack([arr, pad])
    return arr

def main():
    parser = argparse.ArgumentParser(description="SGAP-PPIS Protein Interaction Site Predictor")
    parser.add_argument('--pdb', type=str, required=False, help="Path to PDB file or 4-letter PDB code (e.g. 4bvx)")
    parser.add_argument('--chain', type=str, default='A', help="Chain ID to predict (e.g. A, B)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Trained model checkpoint path")
    parser.add_argument('--model_mode', type=str, default='rectified', choices=['baseline', 'rectified'])
    parser.add_argument('--threshold', type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument('--cutoff', type=float, default=14.0, help="Spatial distance cutoff threshold in Angstroms")
    parser.add_argument('--use_plm', action='store_true', help="Use ESM-2 language model embeddings (Gap 4)")
    parser.add_argument('--feature_dir', type=str, default=None, help="Directory containing precomputed .npy features")
    parser.add_argument('--allow_dummy_features', action='store_true',
                        help="Allow mock geometric features for offline dry-run testing only (issues warning)")
    parser.add_argument('--out_dir', type=str, default='./predictions', help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve PDB file
    pdb_path = args.pdb
    if not pdb_path:
        pdb_path = '4bvx'  # Default demo to 4bvx
    if not os.path.exists(pdb_path) and len(pdb_path) == 4:
        print(f"Downloading PDB structure {pdb_path.upper()}...")
        pdbl = PDBList()
        downloaded = pdbl.retrieve_pdb_file(pdb_path, pdir=args.out_dir, file_format='pdb')
        pdb_path = downloaded

    print(f"Parsing PDB: {pdb_path}, Chain: {args.chain}...")
    seq, res_ids, res_names, pos = parse_pdb_chain(pdb_path, args.chain)
    N = len(seq)
    print(f"Loaded chain with {N} residues: {seq[:30]}...")

    # Centralize coordinates
    pos_centered = pos - np.mean(pos, axis=0, keepdims=True)
    pos_tensor = torch.from_numpy(pos_centered.astype(np.float32))

    # Graph adjacency
    diff = pos_centered[:, np.newaxis, :] - pos_centered[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    adj_mask = (dist_matrix >= 0) & (dist_matrix <= args.cutoff)
    norm_adj = normalize_adjacency(adj_mask.astype(np.float32))
    norm_adj_tensor = torch.from_numpy(norm_adj.astype(np.float32))

    src_nodes, dst_nodes = np.where(adj_mask == 1.0)
    non_self = (src_nodes != dst_nodes)
    src_nodes, dst_nodes = src_nodes[non_self], dst_nodes[non_self]
    edge_index = torch.tensor(np.stack([src_nodes, dst_nodes], axis=0), dtype=torch.long)

    # Edge features
    edge_dists = dist_matrix[src_nodes, dst_nodes] / args.cutoff
    pos_src = pos_centered[src_nodes]
    pos_dst = pos_centered[dst_nodes]
    norm_src = np.linalg.norm(pos_src, axis=-1, keepdims=True) + 1e-8
    norm_dst = np.linalg.norm(pos_dst, axis=-1, keepdims=True) + 1e-8
    cos_sim = np.sum((pos_src / norm_src) * (pos_dst / norm_dst), axis=-1)
    norm_cos = (cos_sim + 1.0) / 2.0
    edge_feats = np.stack([edge_dists, norm_cos], axis=1).astype(np.float32)
    edge_attr_tensor = torch.from_numpy(edge_feats)

    # 62D node features
    ppe = np.linalg.norm(pos_centered, axis=-1, keepdims=True) / 15.0  # (N, 1)

    # Locate precomputed features
    base_pdb = os.path.basename(pdb_path).lower()
    if base_pdb.startswith('pdb'):
        code = base_pdb[3:7]
    else:
        code = base_pdb[:4]
    pid = f"{code}{args.chain}"
    possible_feat_dirs = [
        args.feature_dir,
        "./Feature",
        "../gte-ppis/Feature"
    ]
    feat_base = None
    for f_dir in possible_feat_dirs:
        if f_dir and os.path.exists(os.path.join(f_dir, "dssp", f"{pid}.npy")):
            feat_base = f_dir
            break

    plm_tensor = None
    if feat_base is not None:
        print(f"Loading precomputed biophysical features for {pid} from {feat_base}...")
        dssp = align_len(np.load(os.path.join(feat_base, "dssp", f"{pid}.npy")), N)
        pssm = align_len(np.load(os.path.join(feat_base, "pssm", f"{pid}.npy")), N)
        hmm = align_len(np.load(os.path.join(feat_base, "hmm", f"{pid}.npy")), N)
        resaf = align_len(np.load(os.path.join(feat_base, "resAF", f"{pid}.npy")), N)
        x_62 = np.concatenate([pssm, hmm, dssp, resaf, ppe], axis=1).astype(np.float32)

        if args.use_plm:
            esm_path = os.path.join(feat_base, "esm2", f"{pid}.npy")
            if os.path.exists(esm_path):
                esm = align_len(np.load(esm_path).astype(np.float32), N)
                plm_tensor = torch.from_numpy(esm).to(device)
            else:
                plm_tensor = torch.zeros((N, 1280), dtype=torch.float32, device=device)
    else:
        if not args.allow_dummy_features:
            raise FileNotFoundError(
                f"Missing precomputed features for protein '{pid}'.\n"
                f"SGAP-PPIS requires 62D handcrafted features (PSSM, HMM, DSSP, resAF) in `./Feature`.\n"
                f"To test inference pipeline in dry-run mode, pass `--allow_dummy_features` explicitly."
            )
        print("\n" + "!" * 70)
        print(" [WARNING] Running in DRY-RUN mode with mock geometric node features (--allow_dummy_features).")
        print(" Outputs generated are strictly mock demonstrations and not valid biophysical predictions.")
        print("!" * 70 + "\n")
        dummy_evol = np.zeros((N, 40), dtype=np.float32)
        dummy_struct = np.zeros((N, 21), dtype=np.float32)
        x_62 = np.concatenate([dummy_evol, dummy_struct, ppe], axis=1).astype(np.float32)
        if args.use_plm:
            plm_tensor = torch.zeros((N, 1280), dtype=torch.float32, device=device)

    x_tensor = torch.from_numpy(x_62)

    # Load model
    is_rectified = (args.model_mode == 'rectified')
    model = SGAP_PPIS(
        in_dim=62,
        dg=256,
        da=128,
        num_egnn_layers=5,
        num_appnp_layers=5,
        rectified=is_rectified,
        use_plm=args.use_plm
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded trained checkpoint: {args.checkpoint}")
    else:
        ckpt_dir = "./checkpoints"
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt') and f"sgap_{args.model_mode}" in f]
            if ckpts:
                p = os.path.join(ckpt_dir, ckpts[0])
                ckpt = torch.load(p, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded checkpoint: {p}")

    model.eval()
    with torch.no_grad():
        out = model(
            x_tensor.to(device),
            pos_tensor.to(device),
            edge_index.to(device),
            edge_attr_tensor.to(device),
            norm_adj_tensor.to(device),
            plm_feats=plm_tensor
        )
        probs = out['probs'][:, 1].cpu().numpy()
        preds = (probs >= args.threshold).astype(int)

    # Format predictions table
    results_df = pd.DataFrame({
        'Residue_Num': res_ids,
        'Residue_Name': res_names,
        'Amino_Acid': list(seq),
        'Interaction_Prob': np.round(probs, 4),
        'Predicted_Binding': ['Site' if p == 1 else 'Non-site' for p in preds]
    })

    base_name = f"{os.path.basename(pdb_path).split('.')[0]}_{args.chain}"
    csv_out = os.path.join(args.out_dir, f"{base_name}_predictions.csv")
    results_df.to_csv(csv_out, index=False)

    # Generate PyMOL script
    pml_out = os.path.join(args.out_dir, f"{base_name}_viewer.pml")
    binding_residues = [str(r) for r, p in zip(res_ids, preds) if p == 1]

    pml_lines = [
        f"load {pdb_path}",
        f"select target, chain {args.chain}",
        "hide everything, all",
        "show surface, target",
        "color gray85, target",
        "set transparency, 0.15, target"
    ]
    if binding_residues:
        pml_lines.append(f"select binding_sites, chain {args.chain} and resi {'+'.join(binding_residues)}")
        pml_lines.append("color orange, binding_sites")
    pml_lines.extend([
        "zoom target",
        "set ray_shadows, 0",
        f"# Total binding residues predicted: {len(binding_residues)} / {N} ({len(binding_residues)/N*100:.1f}%)"
    ])

    with open(pml_out, 'w') as f:
        f.write("\n".join(pml_lines))

    print(f"\n=================== Prediction Summary ===================")
    print(f"Total Residues: {N}")
    print(f"Predicted Binding Sites: {sum(preds)} ({sum(preds)/N*100:.1f}%)")
    print(f"Prediction CSV: {csv_out}")
    print(f"PyMOL 3D Script: {pml_out}")
    print(results_df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
