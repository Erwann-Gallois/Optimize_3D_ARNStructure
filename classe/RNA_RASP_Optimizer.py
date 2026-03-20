from RNA_Optimizer import RNA_Optimizer
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type
import torch
import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

class RNA_RASP_Optimizer(RNA_Optimizer):
    def __init__(self, type_RASP="all", **kwargs):
        super().__init__(**kwargs)
        self.type_RASP = type_RASP
        # 1. Chargement Potentiels
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            taille_mat, dict_pots = load_rasp_potentials(path)
            self.potential_tensor = torch.zeros(taille_mat, device=self.device)
            for (k, t1, t2, dist), energy in dict_pots.items():
                self.potential_tensor[k, t1, t2, dist] = self.potential_tensor[k, t2, t1, dist] = energy
        
        # 2. Préparation Structure
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        df_atoms['rasp_type'] = df_atoms.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name'], self.type_RASP), axis=1)
        self.df_filtered = df_atoms[df_atoms['rasp_type'] != -1].reset_index(drop=True)
        
        self.prepare_rigid_structure(self.df_filtered)
        self._setup_rasp_pairs()
        self.potential_tensor = self.potential_tensor.detach()
        self.potential_tensor.requires_grad = False

    def _setup_rasp_pairs(self):
        atom_types = torch.tensor(self.df_filtered['rasp_type'].values, dtype=torch.long, device=self.device)
        res_ids = self.df_filtered['residue_number'].values
        i_idx, j_idx = torch.triu_indices(len(self.df_filtered), len(self.df_filtered), offset=1, device=self.device)
        res_tensor = torch.tensor(res_ids, device=self.device)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask = sep > 3
        self.pair_i, self.pair_j = i_idx[mask].to(torch.int32), j_idx[mask].to(torch.int32)
        self.k_vals = torch.clamp(sep[mask] - 1, 0, 5)
        self.t1_vals, self.t2_vals = atom_types[self.pair_i.long()], atom_types[self.pair_j.long()]
        self.min_dist_vdw = ((self.vdw_radii_all[self.pair_i.long()] + 
                     self.vdw_radii_all[self.pair_j.long()]) * 0.85).detach()

    def calculate_detailed_scores(self, coords):
        p1, p2 = coords[self.pair_i.long()], coords[self.pair_j.long()]
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        
        # Vectorisation massive de l'interpolation RASP
        max_dist_idx = self.potential_tensor.size(3) - 1
        d_clamp = torch.clamp(dists, 0.0, float(max_dist_idx)) 
        d0 = torch.floor(d_clamp).long()
        alpha = d_clamp - d0.float()
        d1 = torch.clamp(d0 + 1, max=max_dist_idx)
        
        # Extraction par batch des énergies
        energy0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        energy1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        
        # Calcul de l'énergie finale avec masque de distance maximale
        rasp_score = torch.sum(((1 - alpha) * energy0 + alpha * energy1) * (dists < float(max_dist_idx)).float())

        bb_penalty, clash_penalty = self.calculate_base_penalties(coords, dists)
        return rasp_score, bb_penalty, clash_penalty