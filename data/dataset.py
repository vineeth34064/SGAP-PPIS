import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PPISDataset(Dataset):
    """
    Dataset loader for Protein-Protein Interaction Sites (PPIS) benchmarks.
    Follows GraphPPIS, ASCE-PPIS, MGMA-PPIS, and SGAP-PPIS benchmark standard:
      - Train_335-1 (334 proteins, excluding 2j3rA)
      - Test_60 (60 proteins)
      - Test_315-28 (287 proteins)
      - UBtest_31-6 (25 proteins)
    """
    def __init__(
        self,
        split='Test_60',
        data_dir=None,
        feature_dir=None,
        cutoff=14.0,
        dist_norm=15.0,
        load_plm=False,
        coord_noise_std=0.0
    ):
        super(PPISDataset, self).__init__()
        self.split = split
        self.cutoff = cutoff
        self.dist_norm = dist_norm
        self.load_plm = load_plm
        self.coord_noise_std = coord_noise_std

        # Locate dataset directory
        possible_data_dirs = [
            data_dir,
            os.path.join(os.path.dirname(__file__), '..', '..', 'gte-ppis', 'Dataset'),
            './Dataset',
            '../gte-ppis/Dataset'
        ]
        self.data_dir = None
        for d in possible_data_dirs:
            if d and os.path.exists(d):
                self.data_dir = d
                break
        if self.data_dir is None:
            raise FileNotFoundError(f"Could not find Dataset directory in: {possible_data_dirs}")

        # Locate feature directory
        possible_feat_dirs = [
            feature_dir,
            os.path.join(os.path.dirname(__file__), '..', '..', 'gte-ppis', 'Feature'),
            './Feature',
            '../gte-ppis/Feature'
        ]
        self.feature_dir = None
        for f in possible_feat_dirs:
            if f and os.path.exists(f):
                self.feature_dir = f
                break
        if self.feature_dir is None:
            raise FileNotFoundError(f"Could not find Feature directory in: {possible_feat_dirs}")

        # Resolve split file name and psepos file name
        split_to_file = {
            'Train_335-1': ('Train_335.pkl', 'Train335_psepos_SC.pkl'),
            'Train_335': ('Train_335.pkl', 'Train335_psepos_SC.pkl'),
            'Test_60': ('Test_60.pkl', 'Test60_psepos_SC.pkl'),
            'Test_315-28': ('Test_315-28.pkl', 'Test315-28_psepos_SC.pkl'),
            'UBtest_31-6': ('UBtest_31-6.pkl', 'UBtest31-6_psepos_SC.pkl'),
        }

        if split not in split_to_file:
            raise ValueError(f"Unknown split: {split}. Allowed: {list(split_to_file.keys())}")

        pkl_name, psepos_name = split_to_file[split]
        dataset_path = os.path.join(self.data_dir, pkl_name)
        psepos_path = os.path.join(self.feature_dir, 'psepos', psepos_name)

        with open(dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        with open(psepos_path, 'rb') as f:
            self.psepos_dict = pickle.load(f)

        # Drop 2j3rA if Train_335-1 to match benchmark exactly
        self.protein_ids = []
        self.sequences = []
        self.labels = []

        for pid, item in raw_data.items():
            if split in ['Train_335-1', 'Train_335'] and pid == '2j3rA':
                # 2j3rA excluded in Train_335-1 benchmark setting
                continue
            seq = item[0]
            lbl = np.array(item[1], dtype=np.int64)
            self.protein_ids.append(pid)
            self.sequences.append(seq)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.protein_ids)

    def _normalize_adjacency(self, adj):
        """
        Symmetric normalization of adjacency with self-loops:
        A_hat = D^(-1/2) (A + I_N) D^(-1/2)
        """
        N = adj.shape[0]
        adj_loop = adj + np.eye(N, dtype=np.float32)
        rowsum = adj_loop.sum(axis=1)
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diag(r_inv)
        norm_adj = r_mat_inv @ adj_loop @ r_mat_inv
        return torch.from_numpy(norm_adj.astype(np.float32))

    def __getitem__(self, index):
        pid = self.protein_ids[index]
        seq = self.sequences[index]
        lbl = self.labels[index]
        N = len(seq)

        # 1. Coordinates (Side-chain centroids)
        raw_pos = self.psepos_dict[pid]
        if raw_pos.shape[0] != N:
            if raw_pos.shape[0] > N:
                raw_pos = raw_pos[:N]
            else:
                pad = np.zeros((N - raw_pos.shape[0], 3), dtype=raw_pos.dtype)
                raw_pos = np.vstack([raw_pos, pad])

        # Centralization (Section II-B: "relative coordinates ... after centralization")
        pos_centered = raw_pos - np.mean(raw_pos, axis=0, keepdims=True)

        # Coordinate jitter regularization (Gap 8 rectification: dynamic robustness)
        if self.coord_noise_std > 0.0:
            noise = np.random.normal(0.0, self.coord_noise_std, size=pos_centered.shape)
            pos_centered = pos_centered + noise

        pos_tensor = torch.from_numpy(pos_centered.astype(np.float32))

        # 2. Handcrafted Node Features: PSSM(20), HMM(20), DSSP(14), resAF(7), PPE(1) -> 62D
        dssp_path = os.path.join(self.feature_dir, 'dssp', f"{pid}.npy")
        pssm_path = os.path.join(self.feature_dir, 'pssm', f"{pid}.npy")
        hmm_path = os.path.join(self.feature_dir, 'hmm', f"{pid}.npy")
        resaf_path = os.path.join(self.feature_dir, 'resAF', f"{pid}.npy")

        dssp = np.load(dssp_path) if os.path.exists(dssp_path) else np.zeros((N, 14), dtype=np.float32)
        pssm = np.load(pssm_path) if os.path.exists(pssm_path) else np.zeros((N, 20), dtype=np.float32)
        hmm = np.load(hmm_path) if os.path.exists(hmm_path) else np.zeros((N, 20), dtype=np.float32)
        resaf = np.load(resaf_path) if os.path.exists(resaf_path) else np.zeros((N, 7), dtype=np.float32)

        # Length alignment
        for arr_name, arr in [('dssp', dssp), ('pssm', pssm), ('hmm', hmm), ('resaf', resaf)]:
            if arr.shape[0] != N:
                if arr.shape[0] > N:
                    arr = arr[:N]
                else:
                    pad = np.zeros((N - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    arr = np.vstack([arr, pad])
                if arr_name == 'dssp': dssp = arr
                elif arr_name == 'pssm': pssm = arr
                elif arr_name == 'hmm': hmm = arr
                elif arr_name == 'resaf': resaf = arr

        # PPE: pseudo-position embedding: Euclidean norm of centered coordinates / dist_norm
        ppe = np.linalg.norm(pos_centered, axis=-1, keepdims=True) / self.dist_norm  # (N, 1)

        # Concatenate 62D node feature vector: [PSSM, HMM, DSSP, AF, PPE] (Equation 1)
        node_feats = np.concatenate([pssm, hmm, dssp, resaf, ppe], axis=1).astype(np.float32)
        node_feats_tensor = torch.from_numpy(node_feats)

        # Relative Solvent Accessibility (from DSSP column 0) for Gap 7
        rsa_val = dssp[:, 0].astype(np.float32)
        rsa_tensor = torch.from_numpy(rsa_val)

        # 3. Distance Map and Adjacency
        dist_map_path = os.path.join(self.feature_dir, 'distance_map_SC', f"{pid}.npy")
        if os.path.exists(dist_map_path):
            dmap = np.load(dist_map_path)
            if dmap.shape[0] != N or dmap.shape[1] != N:
                diff = pos_centered[:, np.newaxis, :] - pos_centered[np.newaxis, :, :]
                dmap = np.sqrt(np.sum(diff ** 2, axis=-1))
        else:
            diff = pos_centered[:, np.newaxis, :] - pos_centered[np.newaxis, :, :]
            dmap = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Graph connectivity: side-chain centroid distance < cutoff (14 A)
        adj_mask = (dmap >= 0) & (dmap <= self.cutoff)
        adj_matrix = adj_mask.astype(np.float32)

        # Symmetrically normalized adjacency matrix
        norm_adj_tensor = self._normalize_adjacency(adj_matrix)

        # Edge index (src, dst)
        src_nodes, dst_nodes = np.where(adj_matrix == 1.0)
        edge_index = torch.tensor(np.stack([src_nodes, dst_nodes], axis=0), dtype=torch.long)

        # 4. Edge Features (Equation Section II-C3: normalized distance and normalized cosine similarity)
        if len(src_nodes) > 0:
            edge_dists = dmap[src_nodes, dst_nodes] / self.cutoff  # in [0, 1]
            pos_src = pos_centered[src_nodes]
            pos_dst = pos_centered[dst_nodes]
            norm_src = np.linalg.norm(pos_src, axis=-1, keepdims=True) + 1e-8
            norm_dst = np.linalg.norm(pos_dst, axis=-1, keepdims=True) + 1e-8
            cos_sim = np.sum((pos_src / norm_src) * (pos_dst / norm_dst), axis=-1)
            norm_cos = (cos_sim + 1.0) / 2.0  # in [0, 1]
            edge_feats = np.stack([edge_dists, norm_cos], axis=1).astype(np.float32)
        else:
            edge_feats = np.zeros((0, 2), dtype=np.float32)

        edge_attr_tensor = torch.from_numpy(edge_feats)

        # 5. Optional PLM Features (ESM-2 embeddings) for Gap 4
        plm_tensor = None
        if self.load_plm:
            esm_path = os.path.join(self.feature_dir, 'esm2', f"{pid}.npy")
            if os.path.exists(esm_path):
                esm = np.load(esm_path).astype(np.float32)
                if esm.shape[0] != N:
                    if esm.shape[0] > N:
                        esm = esm[:N]
                    else:
                        pad = np.zeros((N - esm.shape[0], esm.shape[1]), dtype=np.float32)
                        esm = np.vstack([esm, pad])
                plm_tensor = torch.from_numpy(esm)
            else:
                plm_tensor = torch.zeros((N, 1280), dtype=torch.float32)

        label_tensor = torch.tensor(lbl, dtype=torch.long)

        return {
            'pid': pid,
            'seq': seq,
            'x': node_feats_tensor,          # (N, 62)
            'pos': pos_tensor,               # (N, 3)
            'edge_index': edge_index,        # (2, M)
            'edge_attr': edge_attr_tensor,   # (M, 2)
            'norm_adj': norm_adj_tensor,     # (N, N)
            'y': label_tensor,               # (N,)
            'rsa': rsa_tensor,               # (N,)
            'plm': plm_tensor                # (N, 1280) if load_plm else None
        }

def ppis_collate_fn(batch):
    """
    Collate function for PPIS data loader.
    For PPIS models with variable length N, proteins are typically processed individually (batch size = 1),
    or concatenated into a batch list.
    """
    if len(batch) == 1:
        return batch[0]
    return batch

