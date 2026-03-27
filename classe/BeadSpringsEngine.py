import torch
import numpy as np
import classe.BaseEngine as BaseEngine
import classe.ModelContext as ModelContext
from biopandas.pdb import PandasPdb

class BeadSpringEngine(BaseEngine):
    """
    Moteur d'optimisation représentant la séquence d'ARN comme une chaîne linéaire de 'billes' 
    (chacune correspondant à un nucléotide), soumise à des ressorts (loi de Hooke/FENE).
    Généralement utilisé pour un repliement 3D "gros grain".
    """
    def __init__(self, sequence: str, **kwargs):
        """
        Initialise le système de billes à partir de la séquence nucléotidique.
        L'espacement et la rigidité sont prédéfinis.
        """
        super().__init__(**kwargs)
        self.sequence = sequence.upper()
        self.n_beads = len(sequence)
        self.k = 30.0    # Constante de raideur du ressort connectant les billes
        self.l0 = 6.0    # Longueur d'équilibre de base (distance attendue entre nucléotides)
        

    def initiate_first_structure(self):
        """
        Construit l'architecture PyPDB basique (Structure, Modele, Chaîne, Résidus, Atome "C3'")
        de manière parfaitement linéaire le long de l'axe X.
        """
        structure = Structure.Structure("RNA")
        model = Model.Model(0)
        chain = Chain.Chain("A")
        
        for i, nt in enumerate(self.sequence):
            # Création du résidu (nucléotide)
            res = Residue.Residue((" ", i+1, " "), nt, " ")
            # Création de l'atome C3' à sa position x
            atom = Atom.Atom("C3'", [i * distance_moy, 0, 0], 0, 0, " ", "C3'", i+1, "C")
            res.add(atom)
            chain.add(res)
            
        model.add(chain)
        structure.add(model)
        self.structure = structure
        

    def get_context(self) -> ModelContext:
        """
        Retourne l'état courant des billes (coordonnées et matrices de distance)
        encapsulé dans un objet ModelContext pour la passe évaluatrice.
        """
        # Matrice de distances symétrique complète
        dist_matrix = torch.cdist(self.coords, self.coords) + 1e-8
        
        # Distances uniques entre paires d'atomes distincts (partie triangulaire supérieure)
        i, j = torch.triu_indices(self.n_beads, self.n_beads, offset=1, device=self.device)
        return ModelContext(
            coords=self.coords,
            res_ids=self.res_ids,
            res_names=list(self.sequence),
            dist_matrix=dist_matrix,
            pairwise_dists=dist_matrix[i, j],
            # Diagonale supérieure d'offset 1 : distance linéaire entre n et n+1
            dists_adj=torch.diag(dist_matrix, diagonal=1)
        )

    def calculate_physical_penalties(self, ctx: ModelContext):
        """
        Calcule la pénalité physique (ici, uniquement l'énergie potentielle élastique des ressorts
        entre les atomes séquentiellement voisins).
        """
        # delta_r: différence entre la distance observée d'un couple (i, i+1) et la longueur d'équilibre l0
        delta_r = ctx.dists_adj - self.l0
        
        # Potentiel harmonique k/2 * x^2
        e_fene = 0.5 * self.k * torch.sum(delta_r ** 2)
        return e_fene
 
    def get_optimized_df(self) -> pd.DataFrame:
        """Retourne un DataFrame avec les positions finales des billes."""
        coords = self.coords.detach().cpu().numpy()
        df = pd.DataFrame({
            'res_id': self.res_ids.cpu().numpy(),
            'res_name': list(self.sequence),
            'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2]
        })
        return df

    def run_optimization(self, num_cycles=1, max_iter=100, lr=0.5):
        """
        Déploie l'optimisation globale en utilisant l'algorithme L-BFGS pour minimiser l'énergie 
        totale de la séquence d'ARN.
        
        Args:
            num_cycles: Nombre d'itérations L-BFGS extérieures.
            max_iter: Itérations maximales d'optimisation directes par cycle.
            lr: Pas d'apprentissage.
        """
        # Utilisation du paramètre de coordonnées (billes) comme variable cible de l'optimiseur L-BFGS
        optimizer = torch.optim.LBFGS([self.coords], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        
        for cycle in range(num_cycles):
            def closure():
                # On efface les gradients précédents
                optimizer.zero_grad()
                ctx = self.get_context()
                
                # Pertes cumulées: Statistiques (DFIRE/RASP) + Physiques (Ressorts)
                loss = self.calculate_total_loss(ctx) + self.calculate_physical_penalties(ctx)
                
                # Rétropropagation du gradient PyTorch
                loss.backward()
                return loss
                
            optimizer.step(closure)
            final_loss = closure().item()
            
            if self.verbose: 
                print(f"Cycle {cycle+1} | Loss: {final_loss:.4f}")
            if final_loss < self.best_score: 
                self.best_score = final_loss
                
        return self.get_optimized_df(), self.best_score