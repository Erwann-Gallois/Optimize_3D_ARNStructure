from abc import ABC, abstractmethod
import torch
from typing import Optional, List, Dict, Any
from ModelContext import ModelContext

class BaseEngine(torch.nn.Module, ABC):
    """
    Moteur de base (classe abstraite) gérant le processus d'optimisation d'une structure moléculaire.
    Il sert de fondation aux moteurs spécifiques (ex: BeadSpringEngine, RigidEngine) 
    et intègre un registre de modules de score (termes d'énergie).
    """
    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        """
        Initialise le moteur de base.
        
        Args:
            device: Le périphérique matériel (CPU/GPU) sur lequel exécuter les calculs Tensor.
            verbose: Si True, active l'affichage des logs de l'optimisation.
        """
        super().__init__()
        # Sélection automatique de CUDA s'il est disponible
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        # Registre interne stockant les fonctions d'énergie à optimiser
        self._score_registry: List[Dict[str, Any]] = []
        # Score minimal atteint lors de la session d'optimisation
        self.best_score = float('inf')

    def add_score_module(self, module: torch.nn.Module, weight: float = 1.0, name: str = None):
        """
        Ajoute une fonction de score additionnelle (module énergétique) à la minimisation globale.
        
        Args:
            module: Le module PyTorch évaluant l'énergie à un instant T.
            weight: Le coefficient/poids à appliquer pour ce module lors de la sommation.
            name: Nom d'identification (pour les logs).
        """
        name = name or module.__class__.__name__
        self._score_registry.append({"name": name, "fn": module.to(self.device), "weight": weight})
        if self.verbose: print(f"  [+] Score ajouté : {name} (poids={weight})")

    @abstractmethod
    def get_context(self) -> ModelContext:
        """
        Construit et retourne un ModelContext complet décrivant l'état courant de la structure.
        """
        pass

    def calculate_total_loss(self, ctx: ModelContext) -> torch.Tensor:
        """
        Calcule la somme pondérée de toutes les évaluations produites par les modules de score enregistrés.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for entry in self._score_registry:
            total_loss = total_loss + entry["weight"] * entry["fn"](ctx)
        return total_loss

    @abstractmethod
    def run_optimization(self, **kwargs):
        """
        Exécute la boucle d'optimisation (doit être définie par la classe enfant).
        """
        pass