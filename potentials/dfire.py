# potentials/dfire.py
import os
import torch
import numpy as np
from core.base_potential import BasePotential
import click

class DfirePotential(BasePotential):
    def __init__(self, filepath, weight=1.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(weight, device)
        # 1. Chargement des données du fichier .nrg
        self.load_parameters(filepath)
        
        # 2. Construction de la matrice de traduction (définie dans BasePotential)
        # Cela permet au GPU de mapper les atomes instantanément
        self._build_translation_matrix()

    def get_atom_type(self, res_name, atom_name):
        res_name = res_name.strip().upper()
        if res_name.startswith('R'): res_name = res_name[1:] 
        if len(res_name) > 1 and res_name in ['ADE', 'URA', 'CYT', 'GUA']:
            res_name = res_name[0]
        atom_name = atom_name.strip()
        atom_name = atom_name.replace('*', "'")
        if atom_name.startswith('H'): 
            return -1
        dfire_type = res_name + "_" + atom_name
        return self.type_to_idx.get(dfire_type, -1)
    
    def load_parameters(self, filepath):
        dict_pots = {}
        if os.path.exists(filepath):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"The file {filepath} does not exist.")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = np.loadtxt(filepath, usecols=range(2, 30))
                atom_pairs = np.loadtxt(filepath, usecols=(0, 1), dtype=str)
            dict_pots = {tuple(pair): row for pair, row in zip(atom_pairs, data)}
            if self.verbose:
                click.secho(f"DFIRE potentials loaded successfully from {filepath}.", fg='green')
        else:
            if self.verbose:
                click.secho(f"Potential file not found: {filepath}, DFIRE ignored.", fg='red')
                return 0
            self.potential_tensor = None

        # Get all unique atom types
        all_types = set()
        for t1, t2 in dict_pots.keys():
            all_types.add(t1)
            all_types.add(t2)
        
        self.sorted_types = sorted(list(all_types))
        self.type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        num_types = len(self.sorted_types)
        num_bins = len(next(iter(dict_pots.values())))
        
        # Tensor creation (num_types, num_types, num_bins)
        self.potential_tensor = torch.zeros((num_types, num_types, num_bins), dtype=torch.float32).to(self.device)
        
        for (t1, t2), values in dict_pots.items():
            if t1 in self.type_to_idx and t2 in self.type_to_idx:
                idx1 = self.type_to_idx[t1]
                idx2 = self.type_to_idx[t2]
                self.potential_tensor[idx1, idx2, :] = torch.tensor(values, dtype=torch.float32)
                self.potential_tensor[idx2, idx1, :] = torch.tensor(values, dtype=torch.float32)
    
    def compute_energy(self, coords, pair_i, pair_j, t1_vals, t2_vals, **kwargs):
        if self.potential_tensor is None or self.weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = coords[pair_i]
        p_j = coords[pair_j]
        
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        step = 0.7
        max_dist = 19.6
        num_bins = self.potential_tensor.size(2)
        
        # Interpolation Catmull-Rom
        d_scaled = dists / step
        d_clamp = torch.clamp(d_scaled, 0.0, float(num_bins - 1.0001))
        
        i = torch.floor(d_clamp).long()
        t = d_clamp - i.float()
        
        # Gestion des bords pour les points de contrôle
        idx0 = torch.clamp(i - 1, min=0)
        idx1 = i
        idx2 = torch.clamp(i + 1, max=num_bins - 1)
        idx3 = torch.clamp(i + 2, max=num_bins - 1)
        
        p0 = self.potential_tensor[t1_vals, t2_vals, idx0]
        p1 = self.potential_tensor[t1_vals, t2_vals, idx1]
        p2 = self.potential_tensor[t1_vals, t2_vals, idx2]
        p3 = self.potential_tensor[t1_vals, t2_vals, idx3]
        
        t2 = t * t
        t3 = t2 * t
        
        f0 = -0.5 * t3 + t2 - 0.5 * t
        f1 = 1.5 * t3 - 2.5 * t2 + 1.0
        f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        f3 = 0.5 * t3 - 0.5 * t2
        
        interp_energy = f0 * p0 + f1 * p1 + f2 * p2 + f3 * p3
        
        # Sigmoid cutoff pour une décroissance douce autour de 19.6A
        cutoff = torch.sigmoid(2.0 * (max_dist - dists))
        dfire_score = torch.sum(interp_energy * cutoff)
        
        return self.weight * dfire_score