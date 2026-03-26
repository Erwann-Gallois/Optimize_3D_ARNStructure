import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type
from classe.BaseRNAOptimizer import BaseRNAOptimizer

class RNA_RASP_Optimizer(BaseRNAOptimizer):
    def __init__(self, nbre_nt_exclu = 2, *args, type_RASP="all", **kwargs):
        self.type_RASP = type_RASP
        self.nbre_nt_exclu = nbre_nt_exclu
        super().__init__(*args, **kwargs)

    def setup_potential(self):
        path = f"potentials/{self.type_RASP}.nrg"
        taille, dict_pots = load_rasp_potentials(path)
        self.potential_tensor = torch.zeros(taille, device=self.device)
        for (k, t1, t2, d), energy in dict_pots.items():
            self.potential_tensor[k, t1, t2, d] = energy
            self.potential_tensor[k, t2, t1, d] = energy

    def filter_atoms(self, df):
        df['rasp_type'] = df.apply(lambda r: get_rasp_type(r['residue_name'], r['atom_name'], self.type_RASP), axis=1)
        return df[df['rasp_type'] != -1].reset_index(drop=True)

    def setup_constraints(self):
        super().setup_constraints()
        # Masque spécifique RASP : sep > 0 (inter-nucléotides ou intra-nucléotide)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
        res_tensor = torch.tensor(self.res_ids, dtype=torch.int32, device=self.device)
        
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_k = sep > self.nbre_nt_exclu
        
        self.pair_i = i_idx[mask_k].to(torch.int32)
        self.pair_j = j_idx[mask_k].to(torch.int32)
        self.clash_i, self.clash_j = self.pair_i, self.pair_j
        
        self.k_vals = torch.clamp(sep[mask_k] - 1, 0, 5).long()
        rasp_types = torch.tensor(self.df_filtered['rasp_type'].values, device=self.device)
        self.t1_vals = rasp_types[self.pair_i.long()]
        self.t2_vals = rasp_types[self.pair_j.long()]

    def calculate_bio_score(self, coords):
        p_i, p_j = coords[self.pair_i.long()], coords[self.pair_j.long()]
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        max_d = self.potential_tensor.size(3) - 1
        d0 = torch.floor(torch.clamp(dists, 0, max_d)).long()
        d1 = torch.clamp(d0 + 1, max=max_d)
        alpha = dists - d0.float()
        
        e0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        e1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        
        # RASP utilise un plateau de correction à 2.7
        return torch.sum(((1-alpha)*e0 + alpha*e1 - 2.7) * (dists < max_d).float())