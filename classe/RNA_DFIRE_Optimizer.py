import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type

class RNA_DFIRE_Optimizer:
    def __init__(self, pdb_path, lr=0.2, output_path="output_rigid.pdb", ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5, backbone_weight=100.0):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device : {self.device}")
        self.best_score = float('inf')
        self.backbone_weight = backbone_weight
        
        # 1. Chargement des potentiels
        self.load_dict_potentials()
        
        # 2. Chargement et préparation de la structure rigide
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

    def load_dict_potentials(self):
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            self.convert_dict_to_tensor(dict_pots)
        else:
            print(f"Fichier de potentiel non trouvé : {path}")

    def convert_dict_to_tensor(self, dict_pots):
        # Récupérer tous les types d'atomes uniques
        all_types = set()
        for t1, t2 in dict_pots.keys():
            all_types.add(t1)
            all_types.add(t2)
        
        self.sorted_types = sorted(list(all_types))
        self.type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        num_types = len(self.sorted_types)
        num_bins = len(next(iter(dict_pots.values())))
        
        # Création du tenseur (num_types, num_types, num_bins)
        self.potential_tensor = torch.zeros((num_types, num_types, num_bins), dtype=torch.float32).to(self.device)
        
        for (t1, t2), values in dict_pots.items():
            if t1 in self.type_to_idx and t2 in self.type_to_idx:
                idx1 = self.type_to_idx[t1]
                idx2 = self.type_to_idx[t2]
                self.potential_tensor[idx1, idx2, :] = torch.tensor(values, dtype=torch.float32)
                self.potential_tensor[idx2, idx1, :] = torch.tensor(values, dtype=torch.float32)

    def convert_pdb_to_rigid_tensors(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        # Filtrage et typage DFIRE
        df_atoms['dfire_type'] = df_atoms.apply(lambda row: get_dfire_type(row['atom_name'], row['residue_name']), axis=1)
        self.df_filtered = df_atoms[df_atoms['dfire_type'] != -1].reset_index(drop=True)
        
        # Vérifier que les types existent dans notre dictionnaire de potentiels
        # On ne garde que les atomes dont le type est connu dans DFIRE
        mask_known = self.df_filtered['dfire_type'].isin(self.type_to_idx.keys())
        self.df_filtered = self.df_filtered[mask_known].reset_index(drop=True)
        
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32).to(self.device)
        res_ids = self.df_filtered['residue_number'].values
        unique_res = np.unique(res_ids)
        
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long).to(self.device)
        
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                ref_idx = mask.idxmax()
                ref_coords_init[i] = raw_coords[ref_idx]
            else:
                first_idx = (self.df_filtered['residue_number'] == res).idxmax()
                ref_coords_init[i] = raw_coords[first_idx]

        # PARAMÈTRES OPTIMISABLES : 
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device))
        
        # CONSTANTE : offsets internes
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # Identification Backbone (pour la contrainte de distance O3'-P)
        self.bb_i_idx, self.bb_j_idx = [], []
        for i in range(len(unique_res) - 1):
            res_curr = unique_res[i]
            res_next = unique_res[i+1]
            idx_o3 = self.df_filtered[(self.df_filtered['residue_number'] == res_curr) & (self.df_filtered['atom_name'] == "O3'")].index
            idx_p = self.df_filtered[(self.df_filtered['residue_number'] == res_next) & (self.df_filtered['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                self.bb_i_idx.append(idx_o3[0])
                self.bb_j_idx.append(idx_p[0])
        
        self.bb_i_idx = torch.tensor(self.bb_i_idx, dtype=torch.long).to(self.device)
        self.bb_j_idx = torch.tensor(self.bb_j_idx, dtype=torch.long).to(self.device)
        self.target_bb_dist = 1.61 
        
        # Préparation des paires pour le calcul du score DFIRE
        # Préparation des paires pour le calcul du score DFIRE
        atom_types = torch.tensor([self.type_to_idx[t] for t in self.df_filtered['dfire_type']], dtype=torch.int32).to(self.device)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
        
        res_tensor = torch.tensor(res_ids, dtype=torch.int32).to(self.device)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_inter = sep > 0
        
        # On utilise int32 pour économiser de la mémoire GPU (supporte jusqu'à 2 milliards d'atomes)
        self.pair_i = i_idx[mask_inter].to(torch.int32)
        self.pair_j = j_idx[mask_inter].to(torch.int32)
        self.t1_vals = atom_types[self.pair_i]
        self.t2_vals = atom_types[self.pair_j]

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

    def get_current_full_coords(self):
        R = self.get_rotation_matrices() 
        R_atoms = R[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_detailed_scores(self, current_full_coords):
        n_pairs = self.pair_i.size(0)
        chunk_size = 5000000 # On traite par blocs de 5 millions de paires pour éviter l'OOM CUDA
        
        dfire_score = torch.tensor(0.0, device=self.device)
        
        max_dist = 19.6
        step = 0.7
        num_bins = self.potential_tensor.size(2)
        
        for start_idx in range(0, n_pairs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_pairs)
            
            p1 = current_full_coords[self.pair_i[start_idx:end_idx].long()]
            p2 = current_full_coords[self.pair_j[start_idx:end_idx].long()]
            
            dists = torch.norm(p1 - p2, dim=1) + 1e-8
            
            # Interpolation linéaire pour le score
            d_scaled = dists / step
            d_clamp = torch.clamp(d_scaled, 0.0, float(num_bins - 1))
            
            d0 = torch.floor(d_clamp).long()
            d1 = torch.clamp(d0 + 1, max=num_bins - 1)
            alpha = d_clamp - d0.float()
            
            t1 = self.t1_vals[start_idx:end_idx]
            t2 = self.t2_vals[start_idx:end_idx]
            
            energy0 = self.potential_tensor[t1.long(), t2.long(), d0]
            energy1 = self.potential_tensor[t1.long(), t2.long(), d1]
            
            interp_energy = (1 - alpha) * energy0 + alpha * energy1
            
            # Masque pour les distances > 19.6
            valid_mask = (dists < max_dist).float()
            dfire_score = dfire_score + torch.sum(interp_energy * valid_mask)

        # Contrainte Backbone
        if len(self.bb_i_idx) > 0:
            p_o3 = current_full_coords[self.bb_i_idx]
            p_p = current_full_coords[self.bb_j_idx]
            bb_dists = torch.norm(p_o3 - p_p, dim=1)
            bb_penalty = self.backbone_weight * torch.sum((bb_dists - self.target_bb_dist)**2)
        else:
            bb_penalty = torch.tensor(0.0, device=self.device)
            
        return dfire_score, bb_penalty

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()
        self.best_score = float('inf')

        print(f"🚀 Début de l'optimisation DFIRE (Adam, {self.num_cycles} cycles de {self.epochs_per_cycle} epochs)...")
        print(f"Nombre de paires d'atomes à traiter : {self.pair_i.size(0):,}")
        
        for cycle in range(self.num_cycles):
            print(f"\n--- Cycle {cycle+1}/{self.num_cycles} ---")
            
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                
                coords = self.get_current_full_coords()
                score, penalty = self.calculate_detailed_scores(coords)
                loss = score + penalty
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref_coords.copy_(self.ref_coords)
                    best_rot_angles.copy_(self.rot_angles)
                    
                if step % 20 == 0 or step == self.epochs_per_cycle - 1:
                    print(f"Epoch {step:3d} | Total: {loss.item():.2f} | DFIRE: {score.item():.2f} | Backbone: {penalty.item():.2f}")
                
                # Nettoyage explicite pour libérer de la mémoire si nécessaire
                del coords, score, penalty, loss
            
            # --- SHAKE ---
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                current_noise_c = self.noise_coords * decay
                current_noise_a = self.noise_angles * decay
                
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref_coords)
                    self.rot_angles.copy_(best_rot_angles)
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * current_noise_c)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * current_noise_a)
                
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
                torch.cuda.empty_cache()

        print("\n Optimisation terminée.")
        with torch.no_grad():
            self.ref_coords.copy_(best_ref_coords)
            self.rot_angles.copy_(best_rot_angles)
            
        self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_full_coords = self.get_current_full_coords().detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_full_coords
        out_ppdb.df["ATOM"]["chain_id"] = "A"
        out_ppdb.to_pdb(path=self.output_path)
        print(f"Fichier sauvegardé : {self.output_path}")