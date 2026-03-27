from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class ModelContext:
    """
    Conteneur de données (Data Class) pour passer l'état courant du modèle, 
    tel que les coordonnées et d'autres informations géométriques,
    aux fonctions de score (comme DFIRE ou RASP) lors de l'optimisation.
    """
    coords: torch.Tensor          # Tenseur (N, 3) contenant les coordonnées XYZ des atomes/billes
    res_ids: torch.Tensor         # Tenseur (N,) contenant l'ID de résidu (ou numéro) pour chaque atome/bille
    atom_names: Optional[List[str]] = None      # Liste optionnelle des noms des atomes
    res_names: Optional[List[str]] = None       # Liste optionnelle des noms des résidus (ex: A, U, G, C)
    dist_matrix: Optional[torch.Tensor] = None  # Matrice complète de distance (N, N)
    pairwise_dists: Optional[torch.Tensor] = None # Distances des paires extraites (selon les indices triu)
    dists_adj: Optional[torch.Tensor] = None    # Distances linéaires entre éléments adjacents (i.e. i et i+1)