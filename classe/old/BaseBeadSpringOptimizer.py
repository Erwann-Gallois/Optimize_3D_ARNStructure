import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional


class BaseBeadSpringOptimizer(ABC, torch.nn.Module):
    """
    Classe de base pour les modèles billes-ressorts d'ARN.

    Miroir de BaseRNAOptimizer, mais pour des modèles basés séquence
    (sans PDB). Gère :
      - l'initialisation des coordonnées
      - les pénalités physiques (WCA + FENE)
      - la boucle d'optimisation L-BFGS avec basin-hopping
      - un registre de fonctions de score dynamiques (add_score_function)

    Les sous-classes concrètes doivent implémenter :
      - setup_potential()    → charger les potentiels externes (RASP, etc.)
      - calculate_bio_score() → calculer le score spécifique au potentiel
    """

    def __init__(
        self,
        sequence: str,
        wca_epsilon: float = 1.0,
        wca_sigma: float = 5.0,
        fene_k: float = 30.0,
        fene_r0: float = 6.0,
        fene_R0: float = 2.0,
        lr: float = 0.5,
        num_cycles: int = 1,
        max_iter_per_cycle: int = 100,
        noise_scale: float = 0.5,
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ):
        # Appel des deux __init__ (ABC ne fait rien, nn.Module initialise)
        torch.nn.Module.__init__(self)

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.sequence = sequence.upper()
        self.n_beads = len(sequence)

        # --- Hyperparamètres d'optimisation ---
        self.lr = lr
        self.num_cycles = num_cycles
        self.max_iter_per_cycle = max_iter_per_cycle
        self.noise_scale = noise_scale
        self.verbose = verbose
        self.best_score = float("inf")

        # --- Paramètres physiques WCA ---
        self.wca_epsilon = float(wca_epsilon)
        self.wca_sigma = float(wca_sigma)
        self.wca_cutoff = (2 ** (1 / 6)) * self.wca_sigma

        # --- Paramètres physiques FENE ---
        self.fene_k = float(fene_k)
        self.fene_r0 = float(fene_r0)
        self.fene_R0 = float(fene_R0)

        # --- Registre des fonctions de score additionnelles ---
        # Entries : {"name": str, "fn": Callable, "weight": float}
        self._score_registry: list[dict] = []

        print(f"Initialisation {self.__class__.__name__} sur : {self.device}")

        # 1. Chargement des potentiels (spécifique à l'enfant)
        self.setup_potential()

        # 2. Initialisation des coordonnées optimisables
        init_coords = self._initialize_coordinates()
        self.coords = torch.nn.Parameter(init_coords.to(self.device).contiguous())

    # =========================================================================
    #  Méthodes abstraites (à implémenter dans les sous-classes)
    # =========================================================================

    @abstractmethod
    def setup_potential(self):
        """
        Charger les fichiers de potentiels et préparer les tenseurs d'énergie.
        Appelé automatiquement dans __init__ avant toute optimisation.
        """
        pass

    @abstractmethod
    def calculate_bio_score(
        self,
        coords: torch.Tensor,
        dist_matrix: torch.Tensor,
        pairwise_dists: torch.Tensor,
        dists_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculer le score spécifique au potentiel (RASP, DFIRE, etc.).

        Parameters
        ----------
        coords          : (n_beads, 3)       – positions courantes des billes
        dist_matrix     : (n_beads, n_beads) – matrice des distances euclidiennes
        pairwise_dists  : (n_pairs,)         – distances triu offset=1
        dists_adj       : (n_beads-1,)       – distances entre billes adjacentes

        Returns
        -------
        Scalaire torch différentiable.
        """
        pass

    # =========================================================================
    #  Initialisation des coordonnées
    # =========================================================================

    def _initialize_coordinates(self) -> torch.Tensor:
        """Place les billes sur une ligne avec un léger bruit gaussien."""
        coords = torch.zeros((self.n_beads, 3), dtype=torch.float32)
        coords[:, 0] = torch.arange(self.n_beads, dtype=torch.float32) * self.fene_r0
        coords += torch.randn_like(coords) * 0.5
        return coords

    # =========================================================================
    #  Registre de fonctions de score
    # =========================================================================

    def add_score_function(
        self,
        fn: Callable,
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Enregistre une fonction de score dans le registre.

        La signature attendue est identique à calculate_bio_score :
            fn(coords, dist_matrix, pairwise_dists, dists_adj) -> torch.Tensor

        Les fonctions du registre sont sommées et ajoutées au score retourné
        par calculate_bio_score() dans forward().
        """
        if name is None:
            name = getattr(fn, "__name__", f"score_{len(self._score_registry)}")
        self._score_registry.append({"name": name, "fn": fn, "weight": float(weight)})
        print(f"  ✓ Fonction de score enregistrée : '{name}' (poids={weight})")

    def remove_score_function(self, name: str):
        """Supprime une fonction de score du registre par son nom."""
        before = len(self._score_registry)
        self._score_registry = [e for e in self._score_registry if e["name"] != name]
        removed = before - len(self._score_registry)
        if removed:
            print(f"  ✗ Fonction de score supprimée : '{name}'")
        else:
            print(f"  ⚠ Aucune fonction nommée '{name}' dans le registre.")

    def list_score_functions(self):
        """Affiche le registre courant des fonctions de score."""
        print(f"\n{'─'*50}")
        print(f"  Registre ({len(self._score_registry)} terme(s) externe(s))")
        print(f"{'─'*50}")
        for i, entry in enumerate(self._score_registry):
            print(f"  [{i}] {entry['name']:<25}  poids = {entry['weight']:.4f}")
        print(f"{'─'*50}\n")

    # =========================================================================
    #  Pénalités physiques de base
    # =========================================================================

    def calculate_penalties(
        self,
        coords: torch.Tensor,
        dist_matrix: torch.Tensor,
        pairwise_dists: torch.Tensor,
        dists_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule les pénalités physiques de la chaîne de billes :
          - Potentiel WCA répulsif (toutes les paires)
          - Ressorts FENE-Fraenkel (paires adjacentes)
        """
        # --- WCA ---
        mask = (pairwise_dists < self.wca_cutoff) & (pairwise_dists > 0)
        d = pairwise_dists[mask]

        e_wca = torch.tensor(0.0, device=self.device)
        if d.numel() > 0:
            sr6 = (self.wca_sigma / d) ** 6
            sr12 = sr6 ** 2
            e_wca = torch.sum(4 * self.wca_epsilon * (sr12 - sr6) + self.wca_epsilon)

        # --- FENE ---
        delta_r = dists_adj - self.fene_r0
        ratio_sq = torch.clamp((delta_r / self.fene_R0) ** 2, max=0.999)
        e_fene = -0.5 * self.fene_k * (self.fene_R0 ** 2) * torch.sum(torch.log(1 - ratio_sq))

        # Pénalité quadratique hors-bornes (robustesse numérique)
        out_of_bounds = (torch.abs(delta_r) >= self.fene_R0).float()
        e_fene = e_fene + torch.sum(out_of_bounds * 1e4 * (delta_r ** 2))

        return e_wca + e_fene

    # =========================================================================
    #  Forward
    # =========================================================================

    def forward(self) -> torch.Tensor:
        """
        Loss totale = pénalités physiques + score bio + somme pondérée du registre.
        """
        dist_matrix = torch.cdist(self.coords, self.coords, p=2) + 1e-8

        i_idx, j_idx = torch.triu_indices(self.n_beads, self.n_beads, offset=1)
        pairwise_dists = dist_matrix[i_idx, j_idx]
        dists_adj = torch.diag(dist_matrix, diagonal=1)

        penalties = self.calculate_penalties(
            self.coords, dist_matrix, pairwise_dists, dists_adj
        )
        bio_score = self.calculate_bio_score(
            self.coords, dist_matrix, pairwise_dists, dists_adj
        )

        # Fonctions de score enregistrées dynamiquement
        extra = torch.tensor(0.0, device=self.device)
        for entry in self._score_registry:
            extra = extra + entry["weight"] * entry["fn"](
                self.coords, dist_matrix, pairwise_dists, dists_adj
            )

        return penalties + bio_score + extra

    # =========================================================================
    #  Optimisation (L-BFGS + basin-hopping)
    # =========================================================================

    def run_optimization(self) -> np.ndarray:
        """
        Minimisation d'énergie par L-BFGS avec basin-hopping.

        Utilise les hyperparamètres définis à la construction :
          self.lr, self.num_cycles, self.max_iter_per_cycle, self.noise_scale
        """
        optimizer = torch.optim.LBFGS(
            [self.coords],
            lr=self.lr,
            max_iter=self.max_iter_per_cycle,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        if self.verbose:
            print(f"\n🚀 Début de l'optimisation L-BFGS ({self.num_cycles} cycle(s), lr={self.lr})...")
            if self._score_registry:
                self.list_score_functions()

        best_coords = self.coords.clone().detach()

        for cycle in range(self.num_cycles):
            if self.verbose:
                print(f"\n--- Cycle {cycle + 1}/{self.num_cycles} ---")

            def closure():
                optimizer.zero_grad()
                loss = self.forward()
                loss.backward()
                return loss

            final_loss = optimizer.step(closure)
            loss_val = final_loss.item() if final_loss is not None else float("nan")

            if loss_val < self.best_score:
                self.best_score = loss_val
                best_coords = self.coords.clone().detach()

            if self.verbose:
                print(f"  Énergie finale : {loss_val:.4f}")

            # Basin-hopping : bruit décroissant entre les cycles
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                with torch.no_grad():
                    self.coords.copy_(
                        best_coords
                        + torch.randn_like(best_coords) * self.noise_scale * decay
                    )

        # Restauration du meilleur état
        with torch.no_grad():
            self.coords.copy_(best_coords)

        if self.verbose:
            print(f"\n✅ Optimisation terminée. Meilleure énergie : {self.best_score:.4f}")

        return self.coords.detach().cpu().numpy()
