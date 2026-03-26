from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class ModelContext:
    """Conteneur pour passer l'état courant du modèle aux fonctions de score."""
    coords: torch.Tensor          # (N, 3)
    res_ids: torch.Tensor         # (N,) ID de résidu pour chaque atome/bille
    atom_names: Optional[List[str]] = None
    res_names: Optional[List[str]] = None
    dist_matrix: Optional[torch.Tensor] = None
    pairwise_dists: Optional[torch.Tensor] = None
    dists_adj: Optional[torch.Tensor] = None