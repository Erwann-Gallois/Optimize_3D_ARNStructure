import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from abc import ABC, abstractmethod

# Importations des parseurs spécifiques
try:
    from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
except ImportError:
    print("Attention: parse_dfire_potentials non trouvé.")

try:
    from parse_rasp_potentials import load_rasp_potentials, get_rasp_type
except ImportError:
    print("Attention: parse_rasp_potentials non trouvé.")

class BaseRNAOptimizer(ABC):
    """
    Classe de base gérant la structure rigide de l'ARN, les contraintes physiques
    et la boucle d'optimisation.
    """
    def __init__(self, pdb_path, lr=0.2, output_path="output_optimized.pdb", 
                 ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, 
                 noise_coords=1.5, noise_angles=0.5, backbone_weight=100.0, clash_weight=50.0, verbose=True):
        
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pdb_path = pdb_path
        self.lr = lr
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.noise_coords = noise_coords
        self.noise_angles = noise_angles
        self.backbone_weight = backbone_weight
        self.clash_weight = clash_weight
        self.best_score = float('inf')
        self.verbose = verbose
        
        print(f"Initialisation Optimizer sur : {self.device}")

        # 1. Chargement des potentiels (spécifique à l'enfant)
        self.setup_potential()
        
        # 2. Préparation de la structure et filtrage
        self.prepare_structure()
        
        # 3. Setup des contraintes et des paires d'interaction
        self.setup_constraints()

    @abstractmethod
    def setup_potential(self):
        """Charger les fichiers de potentiels et préparer les tenseurs d'énergie."""
        pass

    @abstractmethod
    def filter_atoms(self, df_atoms):
        """Filtrer les atomes et attribuer les types spécifiques au potentiel."""
        pass

    @abstractmethod
    def calculate_bio_score(self, coords):
        """Calculer le score spécifique (DFIRE, RASP, etc.)."""
        pass

    def prepare_structure(self):
        """Initialise le modèle rigide (Tenseurs, Paramètres, Offsets)."""
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        # Filtrage spécifique à l'implémentation
        self.df_filtered = self.filter_atoms(df_atoms)
        
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, 
                                 dtype=torch.float32, device=self.device)
        self.res_ids = self.df_filtered['residue_number'].values
        unique_res = np.unique(self.res_ids)
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in self.res_ids], 
                                           dtype=torch.long, device=self.device)
        
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), device=self.device)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                ref_coords_init[i] = raw_coords[mask.idxmax()]
            else:
                ref_coords_init[i] = raw_coords[(self.df_filtered['residue_number'] == res).idxmax()]

        # Paramètres optimisables
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), device=self.device))
        
        # Offsets constants par rapport au centre de rotation (ref_atom)
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        self.unique_res = unique_res

    def setup_constraints(self):
        """Identifie les indices pour le Backbone et définit les rayons VDW."""
        # 1. Indices Backbone (Liaisons chimiques et angles)
        self.bb_i_idx, self.bb_j_idx = [], [] 
        self.p_curr_idx, self.p_next_idx = [], [] 
        self.c3_idx = [] 

        for i in range(len(self.unique_res) - 1):
            res_curr, res_next = self.unique_res[i], self.unique_res[i+1]
            df_curr = self.df_filtered[self.df_filtered['residue_number'] == res_curr]
            df_next = self.df_filtered[self.df_filtered['residue_number'] == res_next]
            
            idx_o3 = df_curr[df_curr['atom_name'] == "O3'"].index
            idx_c3 = df_curr[df_curr['atom_name'] == "C3'"].index
            idx_p_c = df_curr[df_curr['atom_name'] == "P"].index
            idx_p_n = df_next[df_next['atom_name'] == "P"].index
            
            if not idx_o3.empty and not idx_p_n.empty:
                self.bb_i_idx.append(idx_o3[0]); self.bb_j_idx.append(idx_p_n[0])
            if not idx_p_c.empty and not idx_p_n.empty:
                self.p_curr_idx.append(idx_p_c[0]); self.p_next_idx.append(idx_p_n[0])
            if not idx_c3.empty and not idx_o3.empty and not idx_p_n.empty:
                self.c3_idx.append(idx_c3[0])
        
        self.bb_i_idx = torch.tensor(self.bb_i_idx, dtype=torch.long, device=self.device)
        self.bb_j_idx = torch.tensor(self.bb_j_idx, dtype=torch.long, device=self.device)
        self.p_curr_idx = torch.tensor(self.p_curr_idx, dtype=torch.long, device=self.device)
        self.p_next_idx = torch.tensor(self.p_next_idx, dtype=torch.long, device=self.device)
        self.c3_idx = torch.tensor(self.c3_idx, dtype=torch.long, device=self.device)

        # 2. Rayons VDW pour les Clashes
        radii_map = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8}
        atom_names = self.df_filtered['atom_name'].values
        self.vdw_radii = torch.tensor([radii_map.get(n[0], 1.7) for n in atom_names], 
                                     dtype=torch.float32, device=self.device)

    def get_rotation_matrices(self):
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])

        N = self.rot_angles.shape[0]
        def eye_batch(): return torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        
        Rx, Ry, Rz = eye_batch(), eye_batch(), eye_batch()
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = cos_b, sin_b, -sin_b, cos_b
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = cos_g, -sin_g, sin_g, cos_g

        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_current_full_coords(self):
        R = self.get_rotation_matrices()[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R, self.offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_penalties(self, coords):
        """Calcule les pénalités physiques (Backbone + Clashes)."""
        penalty = torch.tensor(0.0, device=self.device)
        
        # 1. Clashes Stériques (utilisant les paires de score_bio si définies, ou clash_i)
        # On utilise généralement sep > 1 pour les clashes
        idx_i, idx_j = self.clash_i.long(), self.clash_j.long()
        p_i, p_j = coords[idx_i], coords[idx_j]
        dists = torch.norm(p_i - p_j, dim=1)
        
        # Rayons VDW dynamiques
        thresholds = (self.vdw_radii[idx_i] + self.vdw_radii[idx_j]) * 0.85
        penalty += torch.sum(torch.clamp(thresholds - dists, min=0.0)**2) * self.clash_weight
        
        # 2. Backbone
        if len(self.bb_i_idx) > 0:
            p_o3, p_p = coords[self.bb_i_idx], coords[self.bb_j_idx]
            dist_o3p = torch.norm(p_o3 - p_p, dim=1)
            penalty += torch.sum((dist_o3p - 1.61)**2) * self.backbone_weight
            
            if len(self.p_curr_idx) > 0:
                dist_pp = torch.norm(coords[self.p_curr_idx] - coords[self.p_next_idx], dim=1)
                penalty += torch.sum((dist_pp - 5.9)**2) * (self.backbone_weight * 0.5)
                
            if len(self.c3_idx) > 0:
                p_c3 = coords[self.c3_idx]
                v1, v2 = p_c3 - p_o3, p_p - p_o3
                cos_ang = torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1)*torch.norm(v2, dim=1) + 1e-8)
                penalty += torch.sum((cos_ang - (-0.5))**2) * 20.0
                
        return penalty

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        best_ref = self.ref_coords.clone().detach()
        best_rot = self.rot_angles.clone().detach()

        print(f"🚀 Début de l'optimisation ({self.num_cycles} cycles)...")

        for cycle in range(self.num_cycles):
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                coords = self.get_current_full_coords()
                
                bio_score = self.calculate_bio_score(coords)
                penalty = self.calculate_penalties(coords)
                loss = bio_score + penalty
                
                loss.backward()
                optimizer.step()

                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref.copy_(self.ref_coords)
                    best_rot.copy_(self.rot_angles)
                
                if step % 50 == 0:
                    print(f"Cycle {cycle+1} | Step {step:3d} | Total: {loss.item():.2f} | Bio: {bio_score.item():.2f} | Penalty: {penalty.item():.2f}")

            # SHAKE (Injection de bruit)
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref + torch.randn_like(best_ref) * self.noise_coords * decay)
                    self.rot_angles.copy_(best_rot + torch.randn_like(best_rot) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        with torch.no_grad():
            self.ref_coords.copy_(best_ref)
            self.rot_angles.copy_(best_rot)
        return self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_coords = self.get_current_full_coords().detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        return out_ppdb, self.best_score