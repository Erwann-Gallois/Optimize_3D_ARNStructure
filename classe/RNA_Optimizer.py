import os
import torch
import numpy as np
import pandas as pd
import psutil
from biopandas.pdb import PandasPdb

# Imports spécifiques aux modules de parsing (supposés existants dans votre environnement)
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class RNA_Optimizer:
    """
    Classe de base pour l'optimisation rigide d'ARN. 
    Gère la structure PDB, les transformations rigides, la boucle d'optimisation et les contraintes physiques.
    """
    def __init__(self, pdb_path, lr=0.2, output_path="output_rigid.pdb", ref_atom="C3'", 
                 num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5, 
                 backbone_weight=100.0, verbose=True):
        
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
        self.verbose = verbose
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_bb_dist = 1.61
        self.best_score = float('inf')
        
        self.VDW_RADII = {'P': 1.8, 'O': 1.5, 'C': 1.7, 'N': 1.55, 'H': 1.2}
        self.DEFAULT_VDW = 1.6
        
        self.potential_tensor = None
        self.bb_o3_idx, self.bb_p_idx = [], []

    def prepare_rigid_structure(self, df_atoms):
        """Initialise les paramètres de translation et de rotation pour chaque nucléotide."""
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32).to(self.device)
        res_ids = self.df_filtered['residue_number'].values
        atom_names = self.df_filtered['atom_name'].values
        elements = [name[0] for name in atom_names]
        radii_vals = [self.VDW_RADII.get(e, self.DEFAULT_VDW) for e in elements]
        unique_res = np.unique(res_ids)
        
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long).to(self.device)
        
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                ref_coords_init[i] = raw_coords[mask.idxmax()]
            else:
                ref_coords_init[i] = raw_coords[(self.df_filtered['residue_number'] == res).idxmax()]

        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device))
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # Identification Backbone
        for i in range(len(unique_res) - 1):
            res_curr, res_next = unique_res[i], unique_res[i+1]
            idx_o3 = self.df_filtered[(self.df_filtered['residue_number'] == res_curr) & (self.df_filtered['atom_name'] == "O3'")].index
            idx_p = self.df_filtered[(self.df_filtered['residue_number'] == res_next) & (self.df_filtered['atom_name'] == "P")].index
            if not idx_o3.empty and not idx_p.empty:
                self.bb_o3_idx.append(idx_o3[0])
                self.bb_p_idx.append(idx_p[0])
        
        self.bb_o3_idx = torch.tensor(self.bb_o3_idx, dtype=torch.long).to(self.device)
        self.bb_p_idx = torch.tensor(self.bb_p_idx, dtype=torch.long).to(self.device)
        self.vdw_radii_all = torch.tensor(radii_vals, dtype=torch.float32).to(self.device)

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

    def calculate_base_penalties(self, coords, dists):
        """Calcule les pénalités communes : Clashes VDW et Continuité du Backbone."""
        # Clash
        clash_penalty = torch.tensor(0.0, device=self.device)
        clash_mask = dists < self.min_dist_vdw
        if clash_mask.any():
            diff = self.min_dist_vdw[clash_mask] - dists[clash_mask]
            clash_penalty = torch.tensor(1000.0, device=self.device) * torch.sum(diff**2)

        # Backbone
        bb_penalty = torch.tensor(0.0, device=self.device)
        if len(self.bb_o3_idx) > 0:
            p_o3, p_p = coords[self.bb_o3_idx], coords[self.bb_p_idx]
            bb_dists = torch.norm(p_o3 - p_p, dim=1)
            bb_penalty = self.backbone_weight * torch.sum((bb_dists - self.target_bb_dist)**2)
            
        return bb_penalty, clash_penalty

    def print_metrics(self, step):
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_alloc, gpu_reserved, gpu_name = 0, 0, "CPU Only"
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        print(f" [Métrique Ep {step}] GPU: {gpu_alloc:.1f}MB/{gpu_reserved:.1f}MB ({gpu_name}) | CPU: {cpu_usage}% | RAM: {ram_usage}%")

    def run_optimization(self):
        print(f"Device : {self.device} | Version PyTorch : {torch.__version__}")
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()

        for cycle in range(self.num_cycles):
            if self.verbose: print(f"\n--- Cycle {cycle+1}/{self.num_cycles} ---")
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad()
                coords = self.get_current_full_coords()
                
                # Calcul des scores (spécifique + commun)
                score, bb_penalty, clash_penalty = self.calculate_detailed_scores(coords)
                loss = score + bb_penalty + clash_penalty
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref_coords.copy_(self.ref_coords)
                    best_rot_angles.copy_(self.rot_angles)
                    
                if self.verbose and (step % 20 == 0 or step == self.epochs_per_cycle - 1):
                    print(f"Epoch {step:3d} | Total: {loss.item():.2f} | Potentiel: {score.item():.2f} | BB: {bb_penalty.item():.2f}")
            
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref_coords).add_(torch.randn_like(self.ref_coords) * self.noise_coords * decay)
                    self.rot_angles.copy_(best_rot_angles).add_(torch.randn_like(self.rot_angles) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
                torch.cuda.empty_cache()

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
        print(f"Fichier sauvegardé : {self.output_path} | Meilleur score : {self.best_score:.2f}")