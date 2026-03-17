import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type

class RNA_C3_Optimizer:
    """
    Optimiseur spécialisé qui calcule l'énergie DFIRE uniquement sur les atomes C3'.
    
    Le principe est de :
    1. Ne considérer que les distances C3'-C3' pour le calcul du score (gain de temps massif).
    2. Maintenir les contraintes de squelette (backbone) via les atomes O3' et P.
    3. Optimiser la position (translation) et l'orientation (rotation) de chaque nucléotide.
    4. Régénérer les coordonnées de TOUS les atomes à la fin en appliquant les transformations trouvées.
    """
    
    def __init__(self, pdb_path, lr=0.2, output_path="output_c3.pdb", ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5, backbone_weight=100.0):
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
        print(f"Utilisation du device : {self.device} (Optimiseur C3' uniquement)")
        
        self.best_score = float('inf')
        
        # 1. Chargement des potentiels DFIRE
        self.load_potentials()
        
        # 2. Préparation des données (Atomes C3' pour le score, O3'/P pour le backbone, et tous pour la régénération)
        self.prepare_structure()

    def load_potentials(self):
        """Charge la matrice DFIRE."""
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            # Conversion simplifiée en tenseur
            all_types = sorted(list(set([t for pairs in dict_pots.keys() for t in pairs])))
            self.type_to_idx = {t: i for i, t in enumerate(all_types)}
            num_types = len(all_types)
            num_bins = len(next(iter(dict_pots.values())))
            
            self.potential_tensor = torch.zeros((num_types, num_types, num_bins), dtype=torch.float32).to(self.device)
            for (t1, t2), values in dict_pots.items():
                idx1, idx2 = self.type_to_idx[t1], self.type_to_idx[t2]
                self.potential_tensor[idx1, idx2, :] = torch.tensor(values, dtype=torch.float32)
                self.potential_tensor[idx2, idx1, :] = torch.tensor(values, dtype=torch.float32)
        else:
            raise FileNotFoundError("Matrice DFIRE introuvable.")

    def prepare_structure(self):
        """Prépare les tenseurs de coordonnées et les offsets."""
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        self.full_df = ppdb.df['ATOM'].copy()
        
        # Attribution des types DFIRE pour tous les atomes (nécessaire pour le filtrage)
        self.full_df['dfire_type'] = self.full_df.apply(lambda row: get_dfire_type(row['atom_name'], row['residue_name']), axis=1)
        
        # Extraction des coordonnées de référence (C3') pour chaque résidu
        res_ids = self.full_df['residue_number'].values
        unique_res = np.unique(res_ids)
        num_nucs = len(unique_res)
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        
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
        # On ne garde que les C3' qui ont un type DFIRE connu
        mask_c3 = (self.full_df['atom_name'] == self.ref_atom) & (self.full_df['dfire_type'].isin(self.type_to_idx.keys()))
        mask_bb = self.full_df['atom_name'].isin(["O3'", "P"])
        
        self.df_active = self.full_df[mask_c3 | mask_bb].copy().reset_index(drop=True)
        active_coords = torch.tensor(self.df_active[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)).to(self.device)
        self.active_atom_to_nuc = torch.tensor([res_to_idx[r] for r in self.df_active['residue_number']], dtype=torch.long).to(self.device)
        self.active_offsets = active_coords - ref_coords_init[self.active_atom_to_nuc]
        
        # Index des paires C3' pour le calcul du score (uniquement C3')
        idx_c3_in_active = self.df_active[self.df_active['atom_name'] == self.ref_atom].index
        
        # On calcule d'abord sur CPU pour pouvoir indexer l'index pandas
        pair_i_cpu, pair_j_cpu = torch.triu_indices(len(idx_c3_in_active), len(idx_c3_in_active), offset=1, device='cpu')
        
        self.pair_i = torch.tensor(idx_c3_in_active[pair_i_cpu], dtype=torch.long).to(self.device)
        self.pair_j = torch.tensor(idx_c3_in_active[pair_j_cpu], dtype=torch.long).to(self.device)
        
        # Types DFIRE pour ces paires
        active_types = [self.type_to_idx[t] for t in self.df_active['dfire_type']]
        self.t1_vals = torch.tensor(active_types, dtype=torch.long).to(self.device)[self.pair_i]
        self.t2_vals = torch.tensor(active_types, dtype=torch.long).to(self.device)[self.pair_j]

        # Préparation Backbone (O3'-P)
        self.bb_i, self.bb_j = [], []
        for i in range(num_nucs - 1):
            res_c, res_n = unique_res[i], unique_res[i+1]
            idx_o3 = self.df_active[(self.df_active['residue_number'] == res_c) & (self.df_active['atom_name'] == "O3'")].index
            idx_p = self.df_active[(self.df_active['residue_number'] == res_n) & (self.df_active['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                self.bb_i.append(idx_o3[0])
                self.bb_j.append(idx_p[0])
        self.bb_i = torch.tensor(self.bb_i, dtype=torch.long).to(self.device)
        self.bb_j = torch.tensor(self.bb_j, dtype=torch.long).to(self.device)
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
        # 1. Score DFIRE (C3' uniquement)
        p1, p2 = coords[self.pair_i], coords[self.pair_j]
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        
        step, num_bins = 0.7, self.potential_tensor.size(2)
        d_scaled = torch.clamp(dists / step, 0.0, float(num_bins - 1))
        d0 = d_scaled.long()
        d1 = torch.clamp(d0 + 1, max=num_bins - 1)
        alpha = d_scaled - d0.float()
        
        e0 = self.potential_tensor[self.t1_vals, self.t2_vals, d0]
        e1 = self.potential_tensor[self.t1_vals, self.t2_vals, d1]
        energy = (1 - alpha) * e0 + alpha * e1
        
        dfire_score = torch.sum(energy * (dists < 19.6).float())
        
        # 2. Backbone penalty
        penalty = torch.tensor(0.0, device=self.device)
        if len(self.bb_i) > 0:
            d_bb = torch.norm(coords[self.bb_i] - coords[self.bb_j], dim=1)
            penalty = self.backbone_weight * torch.sum((d_bb - self.target_bb_dist)**2)
            
        return dfire_score, penalty

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        best_ref = self.ref_coords.clone().detach()
        best_rot = self.rot_angles.clone().detach()
        
        print(f"🚀 Optimisation C3' en cours ({self.num_cycles} cycles)...")
        
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
            
            # Affichage périodique
            print(f"Cycle {cycle+1:2d} | Score: {self.best_score:.2f}")

            # Shake
            if cycle < self.num_cycles - 1:
                with torch.no_grad():
                    decay = 1.0 - (cycle / self.num_cycles)
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * self.noise_coords * decay)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        # Restauration du meilleur état
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
        out_ppdb.df['ATOM'] = self.full_df.copy().drop(columns=['dfire_type'])
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        out_ppdb.to_pdb(path=self.output_path)
        print(f"💾 Structure complète sauvegardée dans : {self.output_path}")
