import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class RNA_RASP_C3_Optimizer:
    """
    Optimiseur spécialisé qui calcule l'énergie RASP uniquement sur les atomes C3'.
    
    Le principe est de :
    1. Ne considérer que les distances C3'-C3' pour le calcul du score (gain de temps massif).
    2. Utiliser la matrice de potentiel RASP spécifique (c3.nrg).
    3. Maintenir les contraintes de squelette (backbone) via les atomes O3' et P.
    4. Optimiser la position (translation) et l'orientation (rotation) de chaque nucléotide.
    5. Régénérer les coordonnées de TOUS les atomes à la fin en appliquant les transformations trouvées.
    """
    
    def __init__(self, pdb_path, lr=0.2, output_path="output_rasp_c3.pdb", ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5, backbone_weight=100.0):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.noise_coords = noise_coords
        self.noise_angles = noise_angles
        self.backbone_weight = backbone_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device : {self.device} (Optimiseur RASP C3' uniquement)")
        
        self.best_score = float('inf')
        
        # 1. Chargement des potentiels RASP C3
        self.load_potentials()
        
        # 2. Préparation de la structure
        self.prepare_structure()

    def load_potentials(self):
        """Charge la matrice RASP C3."""
        path = "potentials/c3.nrg"
        if os.path.exists(path):
            taille_mat, dict_pots = load_rasp_potentials(path)
            # RASP potential_tensor is 4D: [seq_sep, type1, type2, dist_bin]
            self.potential_tensor = torch.zeros(taille_mat, dtype=torch.float32).to(self.device)
            for (k, t1, t2, dist_bin), energy in dict_pots.items():
                if k < taille_mat[0] and t1 < taille_mat[1] and t2 < taille_mat[2] and dist_bin < taille_mat[3]:
                    self.potential_tensor[k, t1, t2, dist_bin] = energy
                    self.potential_tensor[k, t2, t1, dist_bin] = energy
        else:
            raise FileNotFoundError(f"Fichier de potentiel RASP C3 introuvable : {path}")

    def prepare_structure(self):
        """Prépare les tenseurs de coordonnées et les offsets."""
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        self.full_df = ppdb.df['ATOM'].copy()
        
        # Attribution des types RASP pour tous les atomes
        # Pour C3, les types sont 1=A, 2=C, 3=G, 4=U. On soustrait 1 pour l'indexation (0-3).
        self.full_df['rasp_type'] = self.full_df.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name'], "c3"), axis=1)
        self.full_df['rasp_type'] = self.full_df['rasp_type'].apply(lambda x: x - 1 if x != -1 else -1)
        
        # Extraction des résidus uniques
        res_ids = self.full_df['residue_number'].values
        unique_res = np.unique(res_ids)
        num_nucs = len(unique_res)
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        
        # Coordonnées de référence (C3') pour chaque résidu
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device)
        for i, res in enumerate(unique_res):
            mask = (self.full_df['residue_number'] == res) & (self.full_df['atom_name'] == self.ref_atom)
            if mask.any():
                ref_coords_init[i] = torch.tensor(self.full_df.loc[mask.idxmax(), ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)).to(self.device)
            else:
                first_idx = (self.full_df['residue_number'] == res).idxmax()
                ref_coords_init[i] = torch.tensor(self.full_df.loc[first_idx, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)).to(self.device)

        # Paramètres optimisables
        self.ref_coords = torch.nn.Parameter(ref_coords_init)
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device))
        
        # Calcul des offsets pour TOUS les atomes (pour la régénération finale)
        all_coords = torch.tensor(self.full_df[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)).to(self.device)
        self.all_atom_to_nuc = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long).to(self.device)
        self.all_offsets = all_coords - ref_coords_init[self.all_atom_to_nuc]
        
        # Sélection des atomes ACTIFS pour l'optimisation (C3' pour score, O3'/P pour backbone)
        mask_c3 = (self.full_df['atom_name'] == self.ref_atom) & (self.full_df['rasp_type'] != -1)
        mask_bb = self.full_df['atom_name'].isin(["O3'", "P"])
        
        self.df_active = self.full_df[mask_c3 | mask_bb].copy().reset_index(drop=True)
        active_coords = torch.tensor(self.df_active[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)).to(self.device)
        self.active_atom_to_nuc = torch.tensor([res_to_idx[r] for r in self.df_active['residue_number']], dtype=torch.long).to(self.device)
        self.active_offsets = active_coords - ref_coords_init[self.active_atom_to_nuc]
        
        # Index des paires C3' pour le calcul du score
        idx_c3_in_active = self.df_active[self.df_active['atom_name'] == self.ref_atom].index
        
        pair_i_cpu, pair_j_cpu = torch.triu_indices(len(idx_c3_in_active), len(idx_c3_in_active), offset=1, device='cpu')
        
        self.pair_i = torch.tensor(idx_c3_in_active[pair_i_cpu], dtype=torch.long).to(self.device)
        self.pair_j = torch.tensor(idx_c3_in_active[pair_j_cpu], dtype=torch.long).to(self.device)
        
        # Types RASP et indices de résidus pour le calcul de la séparation séquentielle
        self.t1_vals = torch.tensor(self.df_active['rasp_type'].values, dtype=torch.long).to(self.device)[self.pair_i]
        self.t2_vals = torch.tensor(self.df_active['rasp_type'].values, dtype=torch.long).to(self.device)[self.pair_j]
        
        nuc_indices = torch.tensor([res_to_idx[r] for r in self.df_active['residue_number']], dtype=torch.long).to(self.device)
        nuc_i = nuc_indices[self.pair_i]
        nuc_j = nuc_indices[self.pair_j]
        
        # Séparation séquentielle max_sep de la matrice (souvent 9 ici: 0 à 8)
        sep = torch.abs(nuc_i - nuc_j)
        max_k = self.potential_tensor.size(0) - 1
        self.k_vals = torch.clamp(sep - 1, 0, max_k)

        # Préparation Backbone (O3'-P)
        self.bb_o3_idx, self.bb_p_idx = [], []
        for i in range(num_nucs - 1):
            res_c, res_n = unique_res[i], unique_res[i+1]
            idx_o3 = self.df_active[(self.df_active['residue_number'] == res_c) & (self.df_active['atom_name'] == "O3'")].index
            idx_p = self.df_active[(self.df_active['residue_number'] == res_n) & (self.df_active['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                self.bb_o3_idx.append(idx_o3[0])
                self.bb_p_idx.append(idx_p[0])
        self.bb_o3_idx = torch.tensor(self.bb_o3_idx, dtype=torch.long).to(self.device)
        self.bb_p_idx = torch.tensor(self.bb_p_idx, dtype=torch.long).to(self.device)
        self.target_bb_dist = 1.61

    def get_rotation_matrices(self):
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])
        N = self.rot_angles.shape[0]
        Rx = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a
        Ry = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = cos_b, sin_b, -sin_b, cos_b
        Rz = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = cos_g, -sin_g, sin_g, cos_g
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_coords(self, atom_to_nuc, offsets):
        R = self.get_rotation_matrices()[atom_to_nuc]
        rotated_offsets = torch.bmm(R, offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[atom_to_nuc] + rotated_offsets

    def calculate_score(self, coords):
        # 1. Score RASP (C3' uniquement)
        p1, p2 = coords[self.pair_i], coords[self.pair_j]
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        
        # RASP bins: step=0.5, max=20.0 (selon matrice nrg)
        step, num_bins = 0.5, self.potential_tensor.size(3)
        d_scaled = torch.clamp(dists / step, 0.0, float(num_bins - 1))
        d0 = d_scaled.long()
        d1 = torch.clamp(d0 + 1, max=num_bins - 1)
        alpha = d_scaled - d0.float()
        
        # Indexation: [k, t1, t2, dist_bin]
        e0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        e1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        energy = (1 - alpha) * e0 + alpha * e1
        
        rasp_score = torch.sum(energy * (dists < 20.0).float())
        
        # 2. Backbone penalty
        penalty = torch.tensor(0.0, device=self.device)
        if len(self.bb_o3_idx) > 0:
            d_bb = torch.norm(coords[self.bb_o3_idx] - coords[self.bb_p_idx], dim=1)
            penalty = self.backbone_weight * torch.sum((d_bb - self.target_bb_dist)**2)
            
        return rasp_score, penalty

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        best_ref = self.ref_coords.clone().detach()
        best_rot = self.rot_angles.clone().detach()
        
        print(f"🚀 Optimisation RASP C3' en cours ({self.num_cycles} cycles)...")
        
        for cycle in range(self.num_cycles):
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                coords = self.get_coords(self.active_atom_to_nuc, self.active_offsets)
                score, penalty = self.calculate_score(coords)
                loss = score + penalty
                loss.backward()
                optimizer.step()
                
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref.copy_(self.ref_coords)
                    best_rot.copy_(self.rot_angles)
            
            print(f"Cycle {cycle+1:2d} | Meilleur Score: {self.best_score:.2f} | RASP Score : {score.item():.2f} | Pénalité : {penalty.item():.2f}")

            if cycle < self.num_cycles - 1:
                with torch.no_grad():
                    decay = 1.0 - (cycle / self.num_cycles)
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * self.noise_coords * decay)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        with torch.no_grad():
            self.ref_coords.copy_(best_ref)
            self.rot_angles.copy_(best_rot)
        
        print("✅ Optimisation terminée. Régénération de tous les atomes...")
        self.save_pdb()

    def save_pdb(self):
        """Régénère TOUS les atomes à partir des paramètres optimisés."""
        with torch.no_grad():
            final_coords = self.get_coords(self.all_atom_to_nuc, self.all_offsets).cpu().numpy()
        
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.full_df.copy().drop(columns=['rasp_type'])
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        out_ppdb.to_pdb(path=self.output_path)
        print(f"💾 Structure complète sauvegardée dans : {self.output_path}")
