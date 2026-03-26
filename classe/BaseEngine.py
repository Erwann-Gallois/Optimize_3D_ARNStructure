from abc import ABC, abstractmethod
import torch
from typing import Optional, List, Dict, Any
from ModelContext import ModelContext

class BaseEngine(torch.nn.Module, ABC):
    """Moteur de base gérant l'optimisation et le registre de scores."""
    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self._score_registry: List[Dict[str, Any]] = []
        self.best_score = float('inf')

    def add_score_module(self, module: torch.nn.Module, weight: float = 1.0, name: str = None):
        name = name or module.__class__.__name__
        self._score_registry.append({"name": name, "fn": module.to(self.device), "weight": weight})
        if self.verbose: print(f"  [+] Score ajouté : {name} (poids={weight})")

    @abstractmethod
    def get_context(self) -> ModelContext:
        """Doit retourner un ModelContext mis à jour avec les coordonnées courantes."""
        pass

    def calculate_total_loss(self, ctx: ModelContext) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        for entry in self._score_registry:
            total_loss = total_loss + entry["weight"] * entry["fn"](ctx)
        return total_loss

    @abstractmethod
    def run_optimization(self, **kwargs):
        pass