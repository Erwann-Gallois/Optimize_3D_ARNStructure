# core/base_potential.py
import torch
from abc import ABC, abstractmethod
from .topology import NUM_RES, NUM_ATOMS, RESIDUES, ATOMS

class BasePotential(ABC):
    """Interface parente pour tous les potentiels statistiques."""
    
    def __init__(self, weight=1.0, device="cpu", verbose=True):
        self.weight = float(weight)
        self.device = device
        self.verbose = verbose
        self.potential_tensor = None
        
        # Matrice de traduction [4 résidus, 27 atomes]
        self.translation_matrix = torch.full((NUM_RES, NUM_ATOMS), -1, dtype=torch.long, device=self.device)

    def _build_translation_matrix(self):
        """
        Construit la table de conversion Universel -> Spécifique.
        À appeler dans le __init__ des classes filles !
        """
        for res_id, res_name in enumerate(RESIDUES):
            for atom_id, atom_name in enumerate(ATOMS):
                internal_type = self.get_atom_type(res_name, atom_name)
                self.translation_matrix[res_id, atom_id] = internal_type if internal_type is not None else -1

    def map_topology(self, topology):
        """
        Prend la RNATopology et renvoie un tenseur 1D des types spécifiques à ce potentiel.
        C'est cette fonction qui remplace vos boucles 'for' lentes !
        """
        return self.translation_matrix[topology.res_type_ids, topology.atom_type_ids]

    # --- MÉTHODES À REDÉFINIR PAR LES ENFANTS ---

    @abstractmethod
    def get_atom_type(self, res_name, atom_name) -> int:
        """Doit renvoyer l'ID entier spécifique au potentiel (ou -1 si inconnu)."""
        pass

    @abstractmethod
    def compute_energy(self, coords, pair_i, pair_j, t1_vals, t2_vals, **kwargs) -> torch.Tensor:
        """
        Prend l'instance de l'optimiseur (qui contient les coords, pair_i, pair_j)
        et renvoie l'énergie scalaire PyTorch.
        """
        pass