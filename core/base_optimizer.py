# core/base_optimizer.py
import torch
from torch.optim.adam import Adam
import click
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Moteur physique universel. 
    Gère la boucle d'optimisation (Basin Hopping + Adam) de manière agnostique.
    """
    def __init__(
        self, 
        potential, 
        lr=0.2, 
        noise_base=1.5, 
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01,
        verbose=True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.potential = potential
        self.lr = lr
        self.noise_base = noise_base
        self.patience_locale = patience_locale
        self.min_delta = min_delta
        self.patience_globale = patience_globale
        self.taux_refroidissement = taux_refroidissement
        self.bruit_min = bruit_min
        self.verbose = verbose
        
        self.best_score = float('inf')

    # ==========================================
    # MÉTHODES ABSTRAITES (À DÉFINIR DANS LES MODÈLES)
    # ==========================================
    
    @abstractmethod
    def get_optimizable_parameters(self):
        """Retourne la liste des tenseurs PyTorch à optimiser (ex: [coords] ou [ref_coords, rot_angles])"""
        raise NotImplementedError("Subclasses must implement get_optimizable_parameters()")

    @abstractmethod
    def apply_noise(self, current_noise_factor):
        """Applique le bruit spécifique au système lors de la phase de 'Shake'"""
        pass

    @abstractmethod
    def compute_total_energy(self) -> tuple:
        """
        Calcule et retourne l'énergie totale (Énergie interne du modèle + Énergie du potentiel).
        Retourne un tuple: (total_loss, logs_dict) pour l'affichage.
        """
        pass
    
    @abstractmethod
    def save_pdb(self, output_path):
        """Sauvegarde l'état actuel dans un fichier PDB"""
        pass

    # ==========================================
    # GESTION D'ÉTAT UNIVERSELLE
    # ==========================================
    
    def get_current_state(self):
        """Capture l'état actuel de tous les paramètres optimisables (Copie profonde)"""
        return [p.detach().clone() for p in self.get_optimizable_parameters()]

    def set_current_state(self, state_list):
        """Restaure un état précédemment capturé"""
        for p, s in zip(self.get_optimizable_parameters(), state_list):
            p.copy_(s)

    # ==========================================
    # LE MOTEUR (LA BOUCLE D'OPTIMISATION)
    # ==========================================
    
    def run_optimization(self):
        """
        Algorithme de Basin Hopping :
        Alternance entre minimisation locale (Adam) et sauts exploratoires (Shake/Bruit).
        """
        optimizer = Adam(self.get_optimizable_parameters(), lr=self.lr)
        
        best_state = self.get_current_state()
        self.best_score = float('inf')
        
        current_noise = self.noise_base
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            click.secho(f"\n🚀 Démarrage de l'optimisation dynamique sur {self.device}", fg='cyan', bold=True)
            click.secho(f"Potentiel utilisé : {self.potential.__class__.__name__}", fg='magenta')

        # --- BARRE GLOBALE (Exploration) ---
        shake_pbar = tqdm(desc="Global Optimization", unit=" cycles", dynamic_ncols=True)

        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            
            phase_best_score = float('inf')
            phase_best_state = self.get_current_state()
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0

            # --- BARRE LOCALE (Minimisation) ---
            pbar = tqdm(total=10000, desc=f"  └─ Phase {cycle_count}", unit="it", leave=False, dynamic_ncols=True)
            
            # --- DESCENTE DE GRADIENT LOCALE ---
            while True:
                optimizer.zero_grad()
                
                # Le modèle calcule son énergie totale (Interne + Externe)
                result = self.compute_total_energy()
                if result is None:
                    raise RuntimeError("compute_total_energy() returned None. Ensure it returns (total_loss, logs) tuple.")
                total_loss, logs = result
                
                total_loss.backward()
                
                # Optionnel : Gradient clipping pour la stabilité (très utile avec les splines)
                torch.nn.utils.clip_grad_norm_(self.get_optimizable_parameters(), max_norm=5.0)
                
                optimizer.step()

                current_loss = total_loss.item()

                # Mise à jour du meilleur état de CETTE phase locale
                if current_loss < phase_best_score:
                    phase_best_score = current_loss
                    phase_best_state = self.get_current_state()

                # Vérification de la convergence locale (ΔE)
                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0 
                
                prev_loss = current_loss
                epoch += 1

                # Mise à jour de l'affichage
                pbar.update(1)
                if epoch % 100 == 0:
                    logs["Total"] = f"{current_loss:.2f}"
                    pbar.set_postfix(logs)

                # Arrêt si le minimum local est atteint
                if patience_counter >= self.patience_locale or epoch >= 10000:
                    break
            
            pbar.close()

            # --- BILAN DE LA PHASE ---
            if phase_best_score < (self.best_score - self.min_delta):
                if self.verbose:
                    shake_pbar.write(click.style(f"Nouveau record absolu ! {self.best_score:.4f} -> {phase_best_score:.4f}", fg='green', bold=True))
                
                self.best_score = phase_best_score
                best_state = phase_best_state
                cycles_sans_amelioration = 0 
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    shake_pbar.write(click.style(f"Pas d'amélioration. (Échecs: {cycles_sans_amelioration}/{self.patience_globale})", fg='red'))

            shake_pbar.update(1)
            shake_pbar.set_postfix({
                "Best": f"{self.best_score:.2f}", 
                "Fail": f"{cycles_sans_amelioration}/{self.patience_globale}",
                "Noise": f"{current_noise:.3f}"
            })

            # --- PRÉPARATION DU CYCLE SUIVANT (SHAKE) ---
            current_noise *= self.taux_refroidissement 
            
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:               
                with torch.no_grad():
                    # On repart TOUJOURS du meilleur état connu globalement
                    self.set_current_state(best_state)
                    # On laisse le modèle appliquer son bruit spécifique
                    self.apply_noise(current_noise)
                
                # On recrée l'optimiseur pour purger l'inertie (les moments de Adam)
                optimizer = Adam(self.get_optimizable_parameters(), lr=self.lr)
                torch.cuda.empty_cache()

        shake_pbar.close()

        # --- FIN DE L'OPTIMISATION ---
        if self.verbose:
            click.secho("\n✅ Optimisation globale terminée !", fg='green', bold=True)
            if current_noise <= self.bruit_min:
                click.secho("Raison : Refroidissement maximal atteint (Bruit minimum).", fg="yellow")
            else:
                click.secho(f"Raison : Patience globale atteinte ({self.patience_globale} échecs consécutifs).", fg="yellow")

        # Restauration finale du meilleur état absolu
        with torch.no_grad():
            self.set_current_state(best_state)