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
        self.best_score = torch.tensor(float('inf'), device=self.device, dtype=torch.float32)
        
        self.VDW_RADII = {'P': 1.8, 'O': 1.5, 'C': 1.7, 'N': 1.55, 'H': 1.2}
        self.DEFAULT_VDW = 1.6
        
        # Pré-allocation des tenseurs pour éviter les reconstructions
        self.eye = torch.eye(3, device=self.device)
        self.potential_tensor = None
        self.bb_o3_idx, self.bb_p_idx = [], []

    def prepare_rigid_structure(self, df_filtered):
        """Version optimisée : calcul vectorisé des offsets et du mapping."""
        self.df_filtered = df_filtered
        coords = torch.tensor(df_filtered[['x_coord', 'y_coord', 'z_coord']].values, 
                              dtype=torch.float32, device=self.device)
        
        res_ids = torch.tensor(df_filtered['residue_number'].values, device=self.device)
        atom_names = np.array(df_filtered['atom_name'].values)
        
        # Mapping rapide des résidus (0 à N-1)
        unique_res, inverse_indices = torch.unique(res_ids, return_inverse=True)
        self.atom_to_nuc_idx = inverse_indices
        num_nucs = unique_res.size(0)

        # Vectorisation de la recherche de l'atome de référence
        # On crée un masque pour l'atome de référence (ex: C3')
        ref_mask = torch.tensor(atom_names == self.ref_atom, device=self.device)
        
        # Pour chaque résidu, on prend le premier index qui match le ref_atom, sinon le premier index tout court
        ref_indices = torch.zeros(num_nucs, dtype=torch.long, device=self.device)
        for i in range(num_nucs):
            res_mask = (self.atom_to_nuc_idx == i)
            match = res_mask & ref_mask
            if match.any():
                ref_indices[i] = torch.where(match)[0][0]
            else:
                ref_indices[i] = torch.where(res_mask)[0][0]

        self.ref_coords = torch.nn.Parameter(coords[ref_indices].clone())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), device=self.device))
        
        # Offsets calculés en une fois
        self.offsets = (coords - self.ref_coords[self.atom_to_nuc_idx]).detach()

        # Backbone : calcul vectorisé des paires (O3' i -> P i+1)
        is_o3 = torch.tensor(atom_names == "O3'", device=self.device)
        is_p = torch.tensor(atom_names == "P", device=self.device)
        
        o3_indices = torch.where(is_o3)[0]
        
        self.bb_o3_idx = []
        self.bb_p_idx = []
        for idx in o3_indices:
            curr_res_id = res_ids[idx]
            # On cherche l'atome P du résidu suivant
            next_p = torch.where(is_p & (res_ids == curr_res_id + 1))[0]
            if len(next_p) > 0:
                self.bb_o3_idx.append(idx)
                self.bb_p_idx.append(next_p[0])
        
        self.bb_o3_idx = torch.tensor(self.bb_o3_idx, dtype=torch.long, device=self.device)
        self.bb_p_idx = torch.tensor(self.bb_p_idx, dtype=torch.long, device=self.device)

        # VDW Radii
        elements = [name[0] for name in atom_names]
        radii_vals = [self.VDW_RADII.get(e, self.DEFAULT_VDW) for e in elements]
        self.vdw_radii_all = torch.tensor(radii_vals, dtype=torch.float32, device=self.device)
        

    def get_rotation_matrices(self):
        """Calcul batché des matrices de rotation (Euler XYZ)."""
        angles = self.rot_angles
        cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
        sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])
        
        return torch.bmm(self._rz(cz, sz), torch.bmm(self._ry(cy, sy), self._rx(cx, sx)))

    def _rx(self, c, s):
        R = self.eye.repeat(c.shape[0], 1, 1)
        R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = c, -s, s, c
        return R

    def _ry(self, c, s):
        R = self.eye.repeat(c.shape[0], 1, 1)
        R[:, 0, 0], R[:, 0, 2], R[:, 2, 0], R[:, 2, 2] = c, s, -s, c
        return R

    def _rz(self, c, s):
        R = self.eye.repeat(c.shape[0], 1, 1)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
        return R

    def get_current_full_coords(self):
        R = self.get_rotation_matrices() 
        R_atoms = R[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_base_penalties(self, coords, dists):
        """Pénalités calculées entièrement sans sortir du graphe de calcul."""
        # Clash : On évite le any() pour rester sur GPU de manière fluide
        diff = torch.clamp(self.min_dist_vdw - dists, min=0.0)
        clash_penalty = 1000.0 * torch.sum(diff**2)

        # Backbone
        if self.bb_o3_idx.numel() > 0:
            bb_dists = torch.norm(coords[self.bb_o3_idx] - coords[self.bb_p_idx], dim=1)
            bb_penalty = self.backbone_weight * torch.sum((bb_dists - self.target_bb_dist)**2)
        else:
            bb_penalty = torch.tensor(0.0, device=self.device)
            
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
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        print("Device used for optimization:", self.device)
        
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()
        
        for cycle in range(self.num_cycles):
            decay = 1.0 - (cycle / max(1, self.num_cycles - 1))
            if self.verbose:
                print(f"\n--- Cycle {cycle+1}/{self.num_cycles} (Intensité secousse: {decay:.2f}) ---")
            
            for step in range(self.epochs_per_cycle):
                optimizer.zero_grad(set_to_none=True)
                
                coords = self.get_current_full_coords()
                score, bb_penalty, clash_penalty = self.calculate_detailed_scores(coords)
                
                # Correction : On s'assure que le warmup est un scalaire ou un tenseur sans gradient
                warmup_val = 1.0 + 4.0 * torch.exp(torch.tensor(-step / 10.0, device=self.device))
                loss = score + (bb_penalty * warmup_val) + clash_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.ref_coords, self.rot_angles], max_norm=1.0)
                optimizer.step()
                
                # Mise à jour du meilleur score sans .item() pour éviter les synchronisations CPU/GPU
                # On utilise une comparaison de tenseurs sur le device
                if step % 10 == 0: # On ne compare pas à chaque micro-étape pour la vitesse
                    with torch.no_grad():
                        if loss < self.best_score:
                            self.best_score = loss.detach().clone() 
                            best_ref_coords.copy_(self.ref_coords)
                            best_rot_angles.copy_(self.rot_angles)
                    
                if  self.verbose and (step % 50 == 0 or step == self.epochs_per_cycle - 1):
                    # L'affichage reste une synchronisation, mais limité par le verbose
                    print(f" Ep {step:3d} | Total: {loss.detach():.1f}")

            # Secousse vectorisée
            if cycle < self.num_cycles - 1:
                with torch.no_grad():
                    # 1. On repart du meilleur état (déjà détaché normalement)
                    self.ref_coords.copy_(best_ref_coords)
                    self.rot_angles.copy_(best_rot_angles)
                    
                    num_nucs = self.ref_coords.shape[0]
                    nuc_noise_scale = torch.full((num_nucs, 1), 0.2, device=self.device)
                    
                    if self.bb_o3_idx.numel() > 0:
                        # FORCE le detach ici pour casser tout lien résiduel
                        current_coords = self.get_current_full_coords().detach() 
                        p_o3, p_p = current_coords[self.bb_o3_idx], current_coords[self.bb_p_idx]
                        
                        # Calcul de l'erreur sans historique
                        bb_err = torch.abs(torch.norm(p_o3 - p_p, dim=1) - self.target_bb_dist)
                        
                        idx_i = self.atom_to_nuc_idx[self.bb_o3_idx]
                        idx_j = self.atom_to_nuc_idx[self.bb_p_idx]
                        
                        nuc_noise_scale.index_add_(0, idx_i, bb_err.unsqueeze(1) * 2.0)
                        nuc_noise_scale.index_add_(0, idx_j, bb_err.unsqueeze(1) * 2.0)

                    nuc_noise_scale = torch.clamp(nuc_noise_scale, 0.1, 3.0)
                    
                    # On s'assure que le bruit n'a pas de gradient
                    noise_c = torch.randn_like(self.ref_coords) * self.noise_coords * decay
                    noise_a = torch.randn_like(self.rot_angles) * self.noise_angles * decay
                    
                    # Modification in-place des paramètres
                    self.ref_coords.add_(noise_c * nuc_noise_scale)
                    self.rot_angles.add_(noise_a * nuc_noise_scale)
                
                if self.verbose:
                    print(f" > Secousse localisée effectuée.")
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)

        print("\n✅ Optimisation terminée.")
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
        print(f"Fichier sauvegardé : {self.output_path} | Meilleur score : {self.best_score.item():.2f}")