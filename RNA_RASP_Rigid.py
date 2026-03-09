import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
# import depuis votre script
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class RNA_RASP_Rigid:
    def __init__(self, pdb_path, lr=0.2, type_RASP="all", output_path="output_rigid.pdb", ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.type_RASP = type_RASP
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.noise_coords = noise_coords
        self.noise_angles = noise_angles
        
        self.best_score = float('inf')
        
        # 1. Chargement des potentiels
        self.load_dict_potentials()
        
        # 2. Chargement et préparation de la structure rigide
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

    def load_dict_potentials(self):
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            dict_pots = load_rasp_potentials(path)
            self.convert_dict_to_tensor(dict_pots)
        else:
            print(f"Fichier de potentiel non trouvé : {path}")

    def convert_dict_to_tensor(self, dict_pots):
        self.potential_tensor = torch.zeros((6, 23, 23, 20), dtype=torch.float32)
        for (k, t1, t2, dist), energy in dict_pots.items():
            if k < 6 and t1 < 23 and t2 < 23 and dist < 20:
                self.potential_tensor[k, t1, t2, dist] = energy
                self.potential_tensor[k, t2, t1, dist] = energy

    def convert_pdb_to_rigid_tensors(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        df_atoms['rasp_type'] = df_atoms.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name']), axis=1)
        self.df_filtered = df_atoms[df_atoms['rasp_type'] != -1].reset_index(drop=True)
        
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
        res_ids = self.df_filtered['residue_number'].values
        unique_res = np.unique(res_ids)
        
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long)
        
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                ref_idx = mask.idxmax()
                ref_coords_init[i] = raw_coords[ref_idx]
            else:
                first_idx = (self.df_filtered['residue_number'] == res).idxmax()
                ref_coords_init[i] = raw_coords[first_idx]

        # PARAMÈTRES OPTIMISABLES : 
        # 1. Position du centre (translation)
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        
        # 2. Angles de rotation (Euler : rx, ry, rz) initiaux à 0
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32))
        
        # CONSTANTE : offsets internes des atomes par rapport à leur référence
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # Identification Backbone
        self.bb_i_idx, self.bb_j_idx = [], []
        for i in range(len(unique_res) - 1):
            res_curr = unique_res[i]
            res_next = unique_res[i+1]
            idx_o3 = self.df_filtered[(self.df_filtered['residue_number'] == res_curr) & (self.df_filtered['atom_name'] == "O3'")].index
            idx_p = self.df_filtered[(self.df_filtered['residue_number'] == res_next) & (self.df_filtered['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                self.bb_i_idx.append(idx_o3[0])
                self.bb_j_idx.append(idx_p[0])
        
        self.bb_i_idx = torch.tensor(self.bb_i_idx, dtype=torch.long)
        self.bb_j_idx = torch.tensor(self.bb_j_idx, dtype=torch.long)
        self.target_bb_dist = 1.61 
        self.backbone_weight = 500.0 
        
        # Données pour RASP
        atom_types = torch.tensor(self.df_filtered['rasp_type'].values, dtype=torch.long)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1)
        
        res_tensor = torch.tensor(res_ids, dtype=torch.long)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_k = sep > 0 
        
        self.pair_i = i_idx[mask_k]
        self.pair_j = j_idx[mask_k]
        self.k_vals = torch.clamp(sep[mask_k] - 1, 0, 5)
        self.t1_vals = atom_types[self.pair_i]
        self.t2_vals = atom_types[self.pair_j]

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
        # 1. On récupère les matrices de rotation pour chaque nucléotide
        R = self.get_rotation_matrices() 
        
        # 2. On attribue à chaque atome la matrice de son nucléotide
        R_atoms = R[self.atom_to_nuc_idx] # (N_atoms, 3, 3)
        
        # 3. On applique la rotation aux offsets
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        
        # 4. On ajoute les coordonnées de référence (translation)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_detailed_scores(self, current_full_coords):
        p1 = current_full_coords[self.pair_i]
        p2 = current_full_coords[self.pair_j]
        
        # +1e-8 pour éviter le gradient NaN quand la distance est 0
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        
        # Soft mask pour interpolation (pas de changement de taille de tenseur !)
        d_clamp = torch.clamp(dists, 0.0, 18.999) 
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        energy0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        energy1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        
        interp_energy = (1 - alpha) * energy0 + alpha * energy1
        
        # On annule l'énergie des atomes trop éloignés
        valid_mask = (dists < 19.0).float()
        rasp_score = torch.sum(interp_energy * valid_mask)
            
        # Contrainte Backbone
        if len(self.bb_i_idx) > 0:
            p_o3 = current_full_coords[self.bb_i_idx]
            p_p = current_full_coords[self.bb_j_idx]
            bb_dists = torch.norm(p_o3 - p_p, dim=1)
            bb_penalty = self.backbone_weight * torch.sum((bb_dists - self.target_bb_dist)**2)
        else:
            bb_penalty = torch.tensor(0.0)
            
        return rasp_score, bb_penalty

    def run_optimization(self):
        """
        num_cycles : Nombre de fois où on relance l'optimisation après avoir secoué la structure
        epochs_per_cycle : Nombre d'étapes de descente de gradient par cycle
        noise_coords : Amplitude maximale du saut aléatoire en Angströms (translation)
        noise_angles : Amplitude maximale du saut aléatoire en Radians (rotation)
        """
        # Initialisation de l'optimiseur
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()
        self.best_score = float('inf')

        print(f"🚀 Début de l'optimisation (Adam, {self.num_cycles} cycles de {self.epochs_per_cycle} epochs)...")
        
        for cycle in range(self.num_cycles):
            print(f"\n--- Cycle {cycle+1}/{self.num_cycles} ---")
            
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                
                coords = self.get_current_full_coords()
                rasp_score, bb_penalty = self.calculate_detailed_scores(coords)
                loss = rasp_score + bb_penalty
                
                loss.backward()
                optimizer.step()
                
                # Sauvegarde du meilleur état global
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref_coords.copy_(self.ref_coords)
                    best_rot_angles.copy_(self.rot_angles)
                    
                # Affichage
                if step % 20 == 0 or step == self.epochs_per_cycle - 1:
                    print(f"Epoch {step:3d} | Total: {loss.item():.2f} | RASP: {rasp_score.item():.2f} | Backbone: {bb_penalty.item():.2f}")
            
            # --- INJECTION D'ALÉATOIRE (SHAKE) ---
            if cycle < self.num_cycles - 1:
                # Facteur de refroidissement : le bruit diminue au fil des cycles
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                current_noise_c = self.noise_coords * decay
                current_noise_a = self.noise_angles * decay
                
                print(f"Secousse aléatoire ! (Bruit translation: {current_noise_c:.2f}Å, rotation: {current_noise_a:.2f}rad)")
                
                with torch.no_grad():
                    # 1. On repart du meilleur état trouvé pour ne pas perdre nos progrès
                    self.ref_coords.copy_(best_ref_coords)
                    self.rot_angles.copy_(best_rot_angles)
                    
                    # 2. On ajoute un bruit gaussien aléatoire
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * current_noise_c)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * current_noise_a)
                    
                # 3. CRUCIAL : On recrée l'optimiseur pour effacer son inertie passée
                # Sinon Adam va essayer de continuer dans la direction d'avant la secousse
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        print("\n Optimisation terminée, restauration de la meilleure conformation trouvée...")
        with torch.no_grad():
            self.ref_coords.copy_(best_ref_coords)
            self.rot_angles.copy_(best_rot_angles)
            
        self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_full_coords = self.get_current_full_coords().detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_full_coords
        out_ppdb.to_pdb(path=self.output_path)
        print(f"Optimisation Rigide terminée. Meilleur score: {self.best_score:.4f}")
        print(f"Fichier sauvegardé : {self.output_path}")