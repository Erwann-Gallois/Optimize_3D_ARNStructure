# potentials/rasp.py
import os
import torch
from core.base_potential import BasePotential

# Le dictionnaire de typage strict de RASP (repris de votre parseur)
RASP_ATOM_TYPES = {
    **{(res, at): 0 for res in ['A','C','G','U'] for at in ['OP1', 'OP2', 'OP3']},
    **{(res, 'P'): 1 for res in ['A','C','G','U']},
    **{(res, "O5'"): 2 for res in ['A','C','G','U']},
    **{(res, "C5'"): 3 for res in ['A','C','G','U']},
    **{(res, at): 4 for res in ['A','C','G','U'] for at in ["C4'", "C3'", "C2'"]},
    **{(res, at): 5 for res in ['A','C','G','U'] for at in ["O2'", "O3'"]},
    **{(res, "C1'"): 6 for res in ['A','C','G','U']},
    **{(res, "O4'"): 7 for res in ['A','C','G','U']},
    ('C', 'N1'): 8, ('U', 'N1'): 8, ('A', 'N9'): 8, ('G', 'N9'): 8,
    ('A', 'C8'): 9, ('G', 'C8'): 9,
    ('A', 'N3'): 10, ('G', 'N3'): 10, ('A', 'N7'): 10, ('G', 'N7'): 10, ('A', 'N1'): 10, ('C', 'N3'): 10,
    ('A', 'C5'): 11, ('G', 'C5'): 11,
    ('A', 'C4'): 12, ('G', 'C4'): 12,
    ('A', 'C2'): 13,
    ('A', 'C6'): 14, ('C', 'C4'): 14,
    ('A', 'N6'): 15, ('C', 'N4'): 15, ('G', 'N2'): 15,
    ('G', 'C2'): 16,
    ('G', 'C6'): 17, ('U', 'C4'): 17,
    ('C', 'O2'): 18, ('U', 'O2'): 18, ('G', 'O6'): 18, ('U', 'O4'): 18,
    ('C', 'C2'): 19, ('U', 'C2'): 19,
    ('C', 'C5'): 20, ('U', 'C5'): 20,
    ('C', 'C6'): 21, ('U', 'C6'): 21,
    ('G', 'N1'): 22, ('U', 'N3'): 22
}

class RASPPotential(BasePotential):
    def __init__(self, filepath, type_RASP="all", weight=1.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(weight, device)
        self.type_RASP = type_RASP
        self.k_max = 8 if type_RASP == "c3" else 5
        
        # 1. Chargement des données du fichier .nrg
        self.load_parameters(filepath)
        
        # 2. Construction de la matrice de traduction (définie dans BasePotential)
        # Cela permet au GPU de mapper les atomes instantanément
        self._build_translation_matrix()

    def get_atom_type(self, res_name, atom_name):
        """Traduit un atome universel en type RASP (0 à 22)."""
        if self.type_RASP == "c3":
            types = {"A": 0, "C": 1, "G": 2, "U": 3}
            return types.get(res_name, -1)
        
        return RASP_ATOM_TYPES.get((res_name, atom_name), -1)

    def load_parameters(self, filepath):
        """Charge le tenseur d'énergie depuis le fichier .nrg"""
        if not os.path.exists(filepath):
            if self.verbose:
                print(f"Warning: Fichier {filepath} introuvable. RASP ignoré.")
            return

        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Récupération de la taille de la matrice
        taille = lines[1].strip().split("\t")
        taille_mat = tuple(map(int, taille))
        
        self.potential_tensor = torch.zeros(taille_mat, dtype=torch.float32, device=self.device)
        
        header_idx = next(i for i, line in enumerate(lines[:10]) if line.startswith("# K"))
        
        for line in lines[header_idx+1:]:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) == 5:
                k, t1, t2, dist = map(int, parts[:4])
                energy = float(parts[4])
                
                # RASP est symétrique sur t1 et t2
                if k < taille_mat[0] and t1 < taille_mat[1] and t2 < taille_mat[2] and dist < taille_mat[3]:
                    self.potential_tensor[k, t1, t2, dist] = energy
                    self.potential_tensor[k, t2, t1, dist] = energy
                    
        if self.verbose:
            print(f"RASP potentials '{self.type_RASP}' loaded successfully.")

    def compute_energy(self, coords, pair_i, pair_j, t1_vals, t2_vals, **kwargs):
        """
        Calcule l'énergie RASP vectorisée avec interpolation Catmull-Rom.
        seq_sep: Tenseur contenant |res_id_i - res_id_j|
        """
        seq_sep = kwargs.get('seq_sep')
        if seq_sep is None:
            raise ValueError("Le potentiel RASP nécessite l'argument 'seq_sep'.")
        if self.potential_tensor is None or self.weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        # Calcul des distances
        dists = torch.norm(coords[pair_i] - coords[pair_j], dim=1) + 1e-8
        
        # Le facteur k topologique de RASP
        k_vals = torch.clamp(seq_sep, max=self.k_max).long()
        
        max_idx = self.potential_tensor.size(3) - 1
        d_clamp = torch.clamp(dists, 0.5, float(max_idx - 1.5))
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        im1 = torch.clamp(d0 - 1, min=0)
        i2 = torch.clamp(d1 + 1, max=max_idx)
        
        # Récupération des 4 points pour la spline cubique
        p0 = self.potential_tensor[k_vals, t1_vals, t2_vals, im1]
        p1 = self.potential_tensor[k_vals, t1_vals, t2_vals, d0]
        p2 = self.potential_tensor[k_vals, t1_vals, t2_vals, d1]
        p3 = self.potential_tensor[k_vals, t1_vals, t2_vals, i2]
        
        # Interpolation Catmull-Rom
        interp_energy = 0.5 * (
            (2 * p1) + (-p0 + p2) * alpha +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * alpha**2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * alpha**3
        )
        
        # Cutoff lissé autour de 19A
        cutoff = torch.sigmoid(2.0 * (19.0 - dists))
        rasp_score = torch.sum(interp_energy * cutoff)
        
        return self.weight * rasp_score