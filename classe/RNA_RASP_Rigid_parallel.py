import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb

class RNA_RASP_Rigid_parallel:
    def __init__(self, pdb_path, potential_tensor, lr=0.2, output_path=None, 
                 ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, 
                 noise_coords=1.5, noise_angles=0.5, verbose=False):
        
        # Désactiver le parallélisme interne de PyTorch pour laisser le multiprocessing gérer les cœurs
        torch.set_num_threads(1)
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.noise_coords = noise_coords
        self.noise_angles = noise_angles
        self.verbose = verbose
        
        self.best_score = float('inf')
        
        # On utilise le tenseur passé en argument (mémoire partagée)
        self.potential_tensor = potential_tensor
        
        # Chargement et préparation
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

    def convert_pdb_to_rigid_tensors(self, pdb_path):
        """
        Prépare les tenseurs pour l'optimisation rigide avec filtrage 
        automatique des types d'atomes non supportés par le potentiel chargé.
        """
        # 1. Chargement et filtrage initial (atomes connus de get_rasp_type)
        ppdb = PandasPdb().read_pdb(pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        from parse_rasp_potentials import get_rasp_type
        df_atoms['rasp_type'] = df_atoms.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name']), axis=1)
        
        # On ne garde que les atomes qui ont un type RASP valide (>= 0)
        self.df_filtered = df_atoms[df_atoms['rasp_type'] != -1].reset_index(drop=True)
        
        # 2. Extraction des coordonnées et mapping des résidus
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
        res_ids = self.df_filtered['residue_number'].values
        unique_res = np.unique(res_ids)
        
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long)
        
        # 3. Initialisation des centres de référence (Translation)
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                # Utilise l'atome de référence (ex: C3')
                ref_idx = mask.idxmax()
                ref_coords_init[i] = raw_coords[ref_idx]
            else:
                # Fallback sur le premier atome du résidu si l'atome de référence manque
                first_idx = (self.df_filtered['residue_number'] == res).idxmax()
                ref_coords_init[i] = raw_coords[first_idx]

        # Paramètres optimisables
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32))
        
        # Calcul des offsets (coordonnées locales par rapport au centre du nucléotide)
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # 4. Identification des paires pour le Backbone (O3' -> P)
        bb_i, bb_j = [], []
        for i in range(len(unique_res) - 1):
            idx_o3 = self.df_filtered[(self.df_filtered['residue_number'] == unique_res[i]) & 
                                     (self.df_filtered['atom_name'] == "O3'")].index
            idx_p = self.df_filtered[(self.df_filtered['residue_number'] == unique_res[i+1]) & 
                                    (self.df_filtered['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                bb_i.append(idx_o3[0])
                bb_j.append(idx_p[0])
        
        self.bb_i_idx = torch.tensor(bb_i, dtype=torch.long)
        self.bb_j_idx = torch.tensor(bb_j, dtype=torch.long)
        self.target_bb_dist = 1.61 
        self.backbone_weight = 500.0 
        
        # 5. Préparation des paires RASP avec SÉCURITÉ sur les indices
        atom_types = torch.tensor(self.df_filtered['rasp_type'].values, dtype=torch.long)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1)
        
        res_tensor = torch.tensor(res_ids, dtype=torch.long)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_k = sep > 0 # On ne calcule pas l'énergie intra-résidu selon RASP
        
        # Extraction des types pour ces paires
        t1_raw = atom_types[i_idx[mask_k]]
        t2_raw = atom_types[j_idx[mask_k]]
        
        # --- FILTRAGE ANTI-CRASH ---
        # On vérifie la taille du tenseur de potentiel (K, T1, T2, Dist)
        # Si le fichier chargé ne gère que 4 types, max_type_idx sera 3.
        max_type_idx = self.potential_tensor.shape[1] - 1 
        
        # On ne garde que les paires où les DEUX atomes existent dans le potentiel
        valid_mask = (t1_raw <= max_type_idx) & (t2_raw <= max_type_idx)
        
        self.pair_i = i_idx[mask_k][valid_mask]
        self.pair_j = j_idx[mask_k][valid_mask]
        
        # k_vals: sép_séquence - 1, plafonné à la dimension du tenseur (souvent 0-5)
        max_k_idx = self.potential_tensor.shape[0] - 1
        self.k_vals = torch.clamp(sep[mask_k][valid_mask] - 1, 0, max_k_idx)
        
        self.t1_vals = t1_raw[valid_mask]
        self.t2_vals = t2_raw[valid_mask]

        if self.verbose:
            print(f"Structure chargée : {len(unique_res)} résidus, {len(self.pair_i)} paires d'atomes actives.")

    def get_rotation_matrices(self):
        """Convertit les 3 angles d'Euler en une matrice de rotation 3x3 pour chaque nucléotide."""
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])

        N = self.rot_angles.shape[0]
        device = self.rot_angles.device

        Rx = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(device)
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a

        Ry = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(device)
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = cos_b, sin_b, -sin_b, cos_b

        Rz = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(device)
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = cos_g, -sin_g, sin_g, cos_g

        # Matrice finale R = Rz * Ry * Rx
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_current_full_coords(self):
        R_atoms = self.get_rotation_matrices()[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_detailed_scores(self, coords):
        dists = torch.norm(coords[self.pair_i] - coords[self.pair_j], dim=1) + 1e-8
        d_clamp = torch.clamp(dists, 0.0, 18.99)
        d0 = d_clamp.long()
        alpha = d_clamp - d0.float()
        
        e0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        e1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0 + 1]
        
        rasp_score = torch.sum(((1 - alpha) * e0 + alpha * e1) * (dists < 19.0).float())
        
        bb_penalty = torch.tensor(0.0)
        if len(self.bb_i_idx) > 0:
            bb_d = torch.norm(coords[self.bb_i_idx] - coords[self.bb_j_idx], dim=1)
            bb_penalty = self.backbone_weight * torch.sum((bb_d - self.target_bb_dist)**2)
            
        return rasp_score, bb_penalty

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        best_ref = self.ref_coords.clone()
        best_rot = self.rot_angles.clone()

        for cycle in range(self.num_cycles):
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                coords = self.get_current_full_coords()
                rasp, bb = self.calculate_detailed_scores(coords)
                loss = rasp + bb
                loss.backward()
                optimizer.step()
                
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref.copy_(self.ref_coords)
                    best_rot.copy_(self.rot_angles)
            
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / self.num_cycles)
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref + torch.randn_like(best_ref) * self.noise_coords * decay)
                    self.rot_angles.copy_(best_rot + torch.randn_like(best_rot) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        with torch.no_grad():
            self.ref_coords.copy_(best_ref)
            self.rot_angles.copy_(best_rot)
        
        if self.output_path: self.save_optimized_pdb()

    def save_optimized_pdb(self):
        with torch.no_grad():
            final_coords = self.get_current_full_coords().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        out_ppdb.to_pdb(path=self.output_path)