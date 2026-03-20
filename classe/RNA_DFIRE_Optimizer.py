from RNA_Optimizer import RNA_Optimizer
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
import torch
import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

class RNA_DFIRE_Optimizer(RNA_Optimizer):
    def __init__(self, exclusion_index = 2, **kwargs):
        super().__init__(**kwargs)
        self.exclusion_index = exclusion_index
        # 1. Chargement Potentiels
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            self._convert_dict_to_tensor(dict_pots)
        
        # 2. Préparation Structure
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        df_atoms['dfire_type'] = df_atoms.apply(lambda row: get_dfire_type(row['atom_name'], row['residue_name']), axis=1)
        self.df_filtered = df_atoms[df_atoms['dfire_type'] != -1].reset_index(drop=True)
        self.type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        self.df_filtered = self.df_filtered[self.df_filtered['dfire_type'].isin(self.type_to_idx.keys())].reset_index(drop=True)
        
        self.prepare_rigid_structure(self.df_filtered)
        self._setup_dfire_pairs()

    def _convert_dict_to_tensor(self, dict_pots):
        all_types = set()
        for t1, t2 in dict_pots.keys(): all_types.update([t1, t2])
        self.sorted_types = sorted(list(all_types))
        type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        num_bins = len(next(iter(dict_pots.values())))
        self.potential_tensor = torch.zeros((len(self.sorted_types), len(self.sorted_types), num_bins), device=self.device)
        for (t1, t2), values in dict_pots.items():
            idx1, idx2 = type_to_idx[t1], type_to_idx[t2]
            v_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
            self.potential_tensor[idx1, idx2, :] = self.potential_tensor[idx2, idx1, :] = v_tensor

    def _setup_dfire_pairs(self):
        atom_types = torch.tensor([self.type_to_idx[t] for t in self.df_filtered['dfire_type']], dtype=torch.int32, device=self.device)
        res_ids = self.df_filtered['residue_number'].values
        i_idx, j_idx = torch.triu_indices(len(self.df_filtered), len(self.df_filtered), offset=1, device=self.device)
        mask = torch.abs(torch.tensor(res_ids, device=self.device)[i_idx] - torch.tensor(res_ids, device=self.device)[j_idx]) > self.exclusion_index
        self.pair_i, self.pair_j = i_idx[mask].to(torch.int32), j_idx[mask].to(torch.int32)
        self.t1_vals, self.t2_vals = atom_types[self.pair_i.long()], atom_types[self.pair_j.long()]
        self.min_dist_vdw = (self.vdw_radii_all[self.pair_i.long()] + self.vdw_radii_all[self.pair_j.long()]) * 0.85

    def calculate_detailed_scores(self, coords):
        p1, p2 = coords[self.pair_i.long()], coords[self.pair_j.long()]
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        
        # Vectorisation massive de l'interpolation DFIRE
        num_bins = self.potential_tensor.size(2)
        step = 0.7
        d_scaled = dists / step
        d_clamp = torch.clamp(d_scaled, 0.0, float(num_bins - 1))
        d0 = d_clamp.long()
        d1 = torch.clamp(d0 + 1, max=num_bins - 1)
        alpha = (d_clamp - d0.float())
        
        # Extraction par batch des énergies
        energy0 = self.potential_tensor[self.t1_vals.long(), self.t2_vals.long(), d0]
        energy1 = self.potential_tensor[self.t1_vals.long(), self.t2_vals.long(), d1]
        
        # Calcul de l'énergie finale avec masque de distance 19.6A (limite DFIRE)
        dfire_score = torch.sum(((1 - alpha) * energy0 + alpha * energy1) * (dists < 19.6).float())

        bb_penalty, clash_penalty = self.calculate_base_penalties(coords, dists)
        return dfire_score, bb_penalty, clash_penalty