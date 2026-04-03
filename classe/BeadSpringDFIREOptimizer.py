import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO
import subprocess

class BeadSpringDFIREOptimizer:
    def __init__(
        self,
        sequence,
        lr=0.2,
        output_path="output_bead.pdb",
        noise_coords=1.5,
        bead_atom="C3'",
        k=20.0,
        l0=5.5,
        type_RASP="all",
        score_weight=5.0,
        verbose=True,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.output_path = output_path
        self.patience_locale = patience_locale
        self.min_delta = min_delta
        self.patience_globale = patience_globale
        self.taux_refroidissement = taux_refroidissement
        self.bruit_min = bruit_min
        self.noise_coords = float(noise_coords)
        self.bead_atom = bead_atom
        self.best_score = float('inf')
        # Paramètres RASP
        self.type_RASP = type_RASP
        self.dfire_weight = float(score_weight)

        # Paramètres FENE-Fraenkel
        self.k = float(k)
        self.l0 = float(l0)

        self.verbose = verbose

        self.load_dict_potentials()
        self.load_structure(sequence)
        self.setup_pairs()

    def load_dict_potentials(self):
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            self.convert_dict_to_tensor(dict_pots)
            if self.verbose:
                print(f"DFIRE potentials loaded.")
        else:
            if self.verbose:
                print(f"Potential file not found : {path}, DFIRE ignored.")
            self.potential_tensor = None

    def convert_dict_to_tensor(self, dict_pots):
        # Récupérer tous les types d'atomes uniques
        all_types = set()
        for t1, t2 in dict_pots.keys():
            all_types.add(t1)
            all_types.add(t2)
        
        self.sorted_types = sorted(list(all_types))
        self.type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        num_types = len(self.sorted_types)
        num_bins = len(next(iter(dict_pots.values())))
        
        # Création du tenseur (num_types, num_types, num_bins)
        self.potential_tensor = torch.zeros((num_types, num_types, num_bins), dtype=torch.float32).to(self.device)
        
        for (t1, t2), values in dict_pots.items():
            if t1 in self.type_to_idx and t2 in self.type_to_idx:
                idx1 = self.type_to_idx[t1]
                idx2 = self.type_to_idx[t2]
                self.potential_tensor[idx1, idx2, :] = torch.tensor(values, dtype=torch.float32)
                self.potential_tensor[idx2, idx1, :] = torch.tensor(values, dtype=torch.float32)

    def load_structure(self, sequence):
        self.num_beads = len(sequence)
        bead_coords = []
        bead_rows = []

        for i, nt in enumerate(sequence):
            # Coordonnées droites sur x, espacées de l0
            coord = [float(i * self.l0), 0.0, 0.0]
            bead_coords.append(coord)
            
            # On stocke les infos minimales nécessaires pour setup_pairs et save_pdb
            row = {
                'residue_name': nt,
                'residue_number': i + 1,
                'chain_id': 'A',
                'element': self.bead_atom[0],
                'atom_name': self.bead_atom
            }
            bead_rows.append(row)

        bead_coords = np.asarray(bead_coords, dtype=np.float32)
        self.coords = torch.nn.Parameter(torch.tensor(bead_coords, dtype=torch.float32, device=self.device))
        self.template_rows = bead_rows

    def setup_pairs(self):
        i, j = torch.triu_indices(self.num_beads, self.num_beads, offset=1)
        sep = j - i
        
        # paires DFIRE
        if getattr(self, 'potential_tensor', None) is not None:
            mask_dfire = sep > 1  # Separation sequence > 1 pour DFIRE CG
            self.dfire_i = i[mask_dfire].to(self.device)
            self.dfire_j = j[mask_dfire].to(self.device)

            t_vals = []
            for row in self.template_rows:
                res_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
                atom_name = str(row['atom_name']).strip() if 'atom_name' in row else self.bead_atom
                t_str = get_dfire_type(atom_name, res_name)
                
                if t_str in self.type_to_idx:
                    t_vals.append(self.type_to_idx[t_str])
                else:
                    t_vals.append(0) # Type par défaut si inconnu
            
            t_vals_tensor = torch.tensor(t_vals, dtype=torch.long, device=self.device)
            self.t1_vals = t_vals_tensor[self.dfire_i]
            self.t2_vals = t_vals_tensor[self.dfire_j]

    def fene_fraenkel_bond_energy(self):
        """
        Potentiel FENE-Fraenkel sur les liaisons consécutives.

        On pose : delta = r - l0
        E = 1/2 * k * (delta)^2
        """
        p1 = self.coords[:-1]
        p2 = self.coords[1:]

        r = torch.norm(p2 - p1, dim=1) + 1e-8
        delta = r - self.l0

        energy = 0.5 * self.k * (delta ** 2)
        return torch.sum(energy)

    def dfire_like_energy(self):
        if getattr(self, 'potential_tensor', None) is None or self.dfire_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.dfire_i]
        p_j = self.coords[self.dfire_j]
        
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        step = 0.7
        max_dist = 19.6
        num_bins = self.potential_tensor.size(2)
        
        # Interpolation linéaire pour le score
        d_scaled = dists / step
        d_clamp = torch.clamp(d_scaled, 0.0, float(num_bins - 1.0001))
        
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        energy0 = self.potential_tensor[self.t1_vals, self.t2_vals, d0]
        energy1 = self.potential_tensor[self.t1_vals, self.t2_vals, d1]
        
        interp_energy = (1.0 - alpha) * energy0 + alpha * energy1
        
        # Cutoff sigmoid pour une annulation douce autour de 19.6A (comme dans RASP)
        # On peut aussi utiliser un masque brutal: (dists < max_dist).float()
        cutoff = torch.sigmoid(2.0 * (max_dist - dists))
        dfire_score = torch.sum(interp_energy * cutoff)
        
        return self.dfire_weight * dfire_score

    def total_energy(self):
        bond = self.fene_fraenkel_bond_energy()
        dfire = self.dfire_like_energy()
        return bond + dfire, bond, dfire

    def run_optimization(self):
        """
        Optimisation dynamique sans limite fixe d'époques ni de cycles.
        
        - patience_locale : Nb itérations sans amélioration avant de finir une phase de repliement.
        - patience_globale : Nb de "secousses" consécutives sans battre le record absolu avant d'abandonner.
        - taux_refroidissement : Facteur de réduction du bruit à chaque secousse (ex: 0.85).
        - bruit_min : Seuil sous lequel la secousse est considérée comme négligeable.
        """
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        best_coords = self.coords.detach().clone()
        self.best_score = float('inf')
        
        # Le bruit initial remplace le concept de "nombre de cycles"
        current_noise = self.noise_coords
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            print(f"Using device : {self.device}")
            print(f"Starting dynamic optimization (Bassin Hopping/Annealing Algorithm)...")

        # BOUCLE EXTERNE : Contrôlée par le niveau de bruit et les échecs successifs
        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Exploration Phase {cycle_count} (Shake noise: {current_noise:.4f}Å) ---")
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            # BOUCLE INTERNE : Remplace 'for epoch in range(num_epochs)'
            while True:
                optimizer.zero_grad()
                total, bond, dfire = self.total_energy()
                total.backward()
                optimizer.step()

                current_loss = total.item()

                # Condition d'arrêt local (ΔE)
                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0  # On a trouvé une bonne pente, on réinitialise
                
                prev_loss = current_loss
                epoch += 1

                # Affichage tous les 100 pas pour surveiller
                if epoch % 100 == 0:
                    if self.verbose:
                        print(f"  Iteration {epoch:4d} | Total: {current_loss:.4f} | FENE: {bond:.4f} | DFIRE: {dfire:.4f}")

                # Arrêt de la phase locale si on est coincé dans un minimum
                if patience_counter >= self.patience_locale:
                    if self.verbose:
                        print(f"Local minimum reached in {epoch} iterations.")
                    break
                
                # Sécurité : valve de secours pour éviter une boucle infinie pure (ex: oscillation)
                if epoch > 10000:
                    if self.verbose:
                        print(f"Local phase too long, safety cutoff at 10000.")
                    break

            # --- BILAN DU CYCLE ---
            # Est-ce que ce minimum local est le meilleur jamais trouvé ?
            if current_loss < (self.best_score - self.min_delta):
                if self.verbose:
                    print(f"New absolute record ! {self.best_score:.4f} -> {current_loss:.4f}")
                self.best_score = current_loss
                best_coords.copy_(self.coords.detach())
                cycles_sans_amelioration = 0  # On remet le compteur d'échecs à zéro
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    print(f"No record. (Unsuccessful attempts : {cycles_sans_amelioration}/{self.patience_globale})")

            # --- PRÉPARATION DU CYCLE SUIVANT ---
            current_noise *= self.taux_refroidissement  # Le système "refroidit"
            
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                if self.verbose:
                    print(f"Application du SHAKE. Return to the best conformation and add noise.")
                with torch.no_grad():
                    self.coords.copy_(best_coords)
                    self.coords.add_(torch.randn_like(self.coords) * current_noise)
                # On réinitialise l'optimiseur pour effacer la "mémoire" (moment) des gradients précédents
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        # --- FIN DE L'OPTIMISATION ---
        if self.verbose:
            print("\n Global optimization finished !")
            if current_noise <= self.bruit_min:
                print("Reason : The system has cooled (the noise has become too weak to break the bonds).")
            else:
                print(f"Reason : Inability to find a better folding after {self.patience_globale} consecutive shakes.")

        with torch.no_grad():
            self.coords.copy_(best_coords)

        self.save_pdb()

    def save_pdb(self):
        # 1. Récupération des coordonnées optimisées
        coords = self.coords.detach().cpu().numpy()

        # 2. Création de la structure hiérarchique Biopython
        structure = Structure.Structure("optimized")
        model = Model.Model(0)
        structure.add(model)

        # Dictionnaire pour stocker les chaînes créées
        chains = {}

        for i, (row, coord) in enumerate(zip(self.template_rows, coords), start=1):
            # Extraction propre des métadonnées
            chain_id = str(row['chain_id']).strip() if 'chain_id' in row and str(row['chain_id']).strip() else 'A'
            res_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
            res_num = int(row['residue_number']) if 'residue_number' in row else i
            atom_name = str(row['atom_name']).strip() if 'atom_name' in row else self.bead_atom
            
            # Gestion des chaînes
            if chain_id not in chains:
                new_chain = Chain.Chain(chain_id)
                model.add(new_chain)
                chains[chain_id] = new_chain
            
            current_chain = chains[chain_id]

            # Création du résidu (id = (' ', numero, ' '))
            # Note: Biopython utilise un tuple pour l'ID du résidu (hetero-flag, sequence_number, insertion_code)
            residue = Residue.Residue((' ', res_num, ' '), res_name, ' ')
            
            # Création de l'atome
            # Atom.Atom(name, coord, b_factor, occupancy, altloc, fullname, serial_number, element)
            atom = Atom.Atom(
                atom_name, 
                coord.tolist(), 
                0.0,    # B-factor
                1.0,    # Occupancy
                ' ',    # Altloc
                f" {atom_name:<3}", # Fullname (4 caractères avec espaces)
                i,      # Serial number
                element='C' # Element
            )
            
            residue.add(atom)
            current_chain.add(residue)

        # 3. Écriture du fichier avec PDBIO
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.output_path)
        # On sépare chaque espace de votre commande en un élément de liste
        commande = [
            "./../Arena/Arena", 
            self.output_path, 
            self.output_path.replace(".pdb", "_full_atom.pdb"), 
            "5"
        ]

        try:
            # Lancement de l'exécution
            resultat = subprocess.run(commande, capture_output=True, text=True, check=True)
            # Affichage de ce que Arena a renvoyé
            if self.verbose:
                print(f"Well formatted PDB file saved : {self.output_path.replace('.pdb', '_full_atom.pdb')}")
                print(f"Best score : {self.best_score}")
            os.remove(self.output_path)

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print("Error during Arena execution :")
                print(e.stderr)
