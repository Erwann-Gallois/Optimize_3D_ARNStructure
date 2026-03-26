import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
from classe.BaseRNAOptimizer import BaseRNAOptimizer

class RNA_DFIRE_Optimizer(BaseRNAOptimizer):
    def __init__(self, nbre_nt_exclu = 2, *args, **kwargs):
        self.nbre_nt_exclu = nbre_nt_exclu
        super().__init__(*args, **kwargs)
        
    def setup_potential(self):
        path = "potentials/matrice_dfire.dat"
        dict_pots = load_dfire_potentials(path)
        
        all_types = sorted(list(set(t for pair in dict_pots.keys() for t in pair)))
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        num_bins = len(next(iter(dict_pots.values())))
        
        self.potential_tensor = torch.zeros((len(all_types), len(all_types), num_bins), device=self.device)
        for (t1, t2), values in dict_pots.items():
            i1, i2 = self.type_to_idx[t1], self.type_to_idx[t2]
            self.potential_tensor[i1, i2] = torch.tensor(values, device=self.device)
            self.potential_tensor[i2, i1] = torch.tensor(values, device=self.device)

    def filter_atoms(self, df):
        df['dfire_type'] = df.apply(lambda r: get_dfire_type(r['atom_name'], r['residue_name']), axis=1)
        df = df[df['dfire_type'] != -1].reset_index(drop=True)
        return df[df['dfire_type'].isin(self.type_to_idx.keys())].reset_index(drop=True)

    def setup_constraints(self):
        super().setup_constraints()
        # Masque spécifique DFIRE : sep > 2
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
        res_tensor = torch.tensor(self.res_ids, dtype=torch.int32, device=self.device)
        
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_inter = sep > self.nbre_nt_exclu
        
        self.pair_i = i_idx[mask_inter].to(torch.int32)
        self.pair_j = j_idx[mask_inter].to(torch.int32)
        self.clash_i, self.clash_j = self.pair_i, self.pair_j # Utilise les mêmes pour les clashes
        
        # Pré-calcul des types pour accélérer la boucle
        dfire_types = [self.type_to_idx[t] for t in self.df_filtered['dfire_type'].values]
        self.t1_vals = torch.tensor(dfire_types, device=self.device)[self.pair_i.long()]
        self.t2_vals = torch.tensor(dfire_types, device=self.device)[self.pair_j.long()]

    def calculate_bio_score(self, coords):
        p_i, p_j = coords[self.pair_i.long()], coords[self.pair_j.long()]
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        # Interpolation DFIRE
        step = 0.7
        d_scaled = dists / step
        d0 = torch.floor(torch.clamp(d_scaled, 0, 27)).long()
        d1 = torch.clamp(d0 + 1, max=27)
        alpha = d_scaled - d0.float()
        
        e0 = self.potential_tensor[self.t1_vals, self.t2_vals, d0]
        e1 = self.potential_tensor[self.t1_vals, self.t2_vals, d1]
        energy = (1 - alpha) * e0 + alpha * e1
        
        return torch.sum(energy * (dists < 19.6).float())