import torch
import numpy as np

class RNABeadSpringModel(torch.nn.Module):
    def __init__(self, sequence, wca_epsilon=1.0, wca_sigma=5.0, 
                 fene_k=30.0, fene_r0=6.0, fene_R0=2.0, 
                 rasp_weight=1.5, device=None):
        """
        Modèle stérique (billes-ressorts) d'ARN accéléré par PyTorch.
        """
        super().__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device : {self.device}")
        
        self.sequence = sequence.upper()
        self.n_beads = len(sequence)
        
        # --- Paramètres Physiques ---
        self.wca_epsilon = float(wca_epsilon)
        self.wca_sigma = float(wca_sigma)
        self.wca_cutoff = (2**(1/6)) * self.wca_sigma
        
        self.fene_k = float(fene_k)
        self.fene_r0 = float(fene_r0)
        self.fene_R0 = float(fene_R0)
        self.rasp_weight = float(rasp_weight)

        # --- Initialisation des Coordonnées ---
        init_coords = self._initialize_coordinates()
        
        # Le tenseur principal contenant les variables à optimiser 
        # (similaire à self.ref_coords dans RNA_RASP_Optimizer)
        self.coords = torch.nn.Parameter(init_coords.to(self.device).contiguous())

    def _initialize_coordinates(self):
        """Place les billes sur une ligne droite avec un léger bruit gaussien."""
        coords = torch.zeros((self.n_beads, 3), dtype=torch.float32)
        coords[:, 0] = torch.arange(self.n_beads, dtype=torch.float32) * self.fene_r0
        # Ajout d'une petite perturbation pour éviter de rester bloqué dans un optimum local trivial
        coords += torch.randn_like(coords) * 0.5
        return coords

    def energy_wca(self, pairwise_dists):
        """Potentiel de Weeks-Chandler-Andersen (WCA) répulsif en PyTorch."""
        mask = (pairwise_dists < self.wca_cutoff) & (pairwise_dists > 0)
        d = pairwise_dists[mask]
        
        energy = torch.tensor(0.0, device=self.device)
        if len(d) > 0:
            sr6 = (self.wca_sigma / d)**6
            sr12 = sr6**2
            energy = torch.sum(4 * self.wca_epsilon * (sr12 - sr6) + self.wca_epsilon)
            
        return energy

    def energy_fene_fraenkel(self, dists_adj):
        """Ressorts FENE-Fraenkel pour relier les acides nucléiques."""
        delta_r = dists_adj - self.fene_r0
        
        # Le Logarithme de FENE diverge si delta_r >= fene_R0.
        # En PyTorch (via autograd), les 'NaN' se propagent et détruisent l'optimisation.
        # On contraint le ratio pour qu'il soit strictement inférieur à 1.
        ratio_sq = torch.clamp((delta_r / self.fene_R0)**2, max=0.999)
        
        energy = -0.5 * self.fene_k * (self.fene_R0**2) * torch.sum(torch.log(1 - ratio_sq))
        
        # On ajoute une pénalité quadratique stricte et non infinie si jamais une distance 
        # commence son pas d'optimisation (suite à une impulsion forte de descente) trop loin.
        out_of_bounds = (torch.abs(delta_r) >= self.fene_R0).float()
        penalty = torch.sum(out_of_bounds * 1e4 * (delta_r**2))
        
        return energy + penalty

    def score_rasp(self, dist_matrix):
        """
        Placeholder de calcul RASP.
        (A remplacer par la consultation du self.potential_tensor comme dans RNA_RASP_Optimizer).
        """
        score = torch.tensor(0.0, device=self.device)
        
        # Exemple basique différentiable: attractivité pour un repliement artificiel de test (C-G/A-U)
        # i_idx, j_idx = torch.triu_indices(self.n_beads, self.n_beads, offset=4)
        # paires = dist_matrix[i_idx, j_idx]
        # etc.
        
        return score

    def forward(self):
        """Calcule les pseudo-énergies (loss) de toutes les billes du système."""
        # cdist calcule la distance euclidienne entre tous les vecteurs
        # +1e-8 pour éviter le gradient NaN de la norme de torch quand distance == 0
        dist_matrix = torch.cdist(self.coords, self.coords, p=2) + 1e-8
        
        # Extraction du triangle supérieur pour WCA (interactions de toutes les paires)
        i_idx, j_idx = torch.triu_indices(self.n_beads, self.n_beads, offset=1)
        pairwise_dists = dist_matrix[i_idx, j_idx]
        e_wca = self.energy_wca(pairwise_dists)
        
        # Extraction de la diagonale k=1 pour FENE (interactions de la chaîne)
        dists_adj = torch.diag(dist_matrix, diagonal=1)
        e_fene = self.energy_fene_fraenkel(dists_adj)
        
        # RASP pour l'interactome
        e_rasp = self.score_rasp(dist_matrix)
        
        total_energy = e_wca + e_fene + (self.rasp_weight * e_rasp)
        return total_energy

    def run_optimization(self, lr=0.5, cycles=1, max_iter_per_cycle=100):
        """
        Optimisateur L-BFGS pour la minimisation d'énergie.
        """
        # Exige la définition d'un paramètre spécifique dans torch
        optimizer = torch.optim.LBFGS(
            [self.coords],
            lr=lr,
            max_iter=max_iter_per_cycle, # Nb maximal d'itérations L-BFGS par `step()`
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,             # Mémoire H du Hessien pour L-BFGS (entre 5 et 20 habituel)
            line_search_fn="strong_wolfe" # Cherche activement un bon pas (recommandé pour L-BFGS)
        )
        
        print(f"🚀 Début de l'optimisation (L-BFGS, lr={lr})...")
        
        best_loss = float('inf')
        
        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle+1}/{cycles} ---")
            
            # Dans L-BFGS, la méthode 'step()' a impérativement besoin d'une closure() 
            # sans arguments pour recalculer et réeffectuer plusieurs passes en internes.
            def closure():
                optimizer.zero_grad()
                loss = self.forward()
                loss.backward()
                return loss
            
            # Effectue toutes les max_iter_per_cycle de ce cycle
            # et update self.coords de manière cachée sous le capot
            final_loss = optimizer.step(closure)
            
            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                
            print(f"Énergie finale de ce cycle: {final_loss.item():.4f}")
            
        print("Optimisation terminée.")
        return self.coords.detach().cpu().numpy()


# === TEST ===
if __name__ == "__main__":
    seq = "GGCACUUCGGUGCC"
    
    model = RNABeadSpringModel(
        sequence=seq,
        wca_epsilon=1.5,
        wca_sigma=5.5,
        fene_k=30.0,
        fene_r0=6.0,
        fene_R0=2.0,
        rasp_weight=1.5,
        device=torch.device('cpu')
    )
    
    # On la place sur le CPU ou GPU actif automatiquement
    model.to(model.device)
    
    coords3d = model.run_optimization(lr=0.5, cycles=1, max_iter_per_cycle=100)
    
    print("\nCoordonnées optimisées (5 premiers C3') :")
    for i, c in enumerate(coords3d[:5]):
        print(f"C3' {seq[i]}: x={c[0]:.2f}, y={c[1]:.2f}, z={c[2]:.2f}")
