import os
import torch
import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO
import subprocess

# Importation du parseur spécifique pour rsRNASP
from parse_rsrnasp_potentials import load_rsrnasp_potentials, get_rsrnasp_type

class BeadSpringRsRNASPOptimizer:
    def __init__(
        self,
        sequence,
        lr=0.2,
        output_path="output_bead.pdb",
        noise_coords=1.5,
        bead_atom="C3'",
        k=20.0,
        l0=5.5,
        score_weight=1.0,
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
        
        # Paramètres d'arrêt dynamique (Bassin Hopping)
        self.patience_locale = patience_locale
        self.min_delta = min_delta
        self.patience_globale = patience_globale
        self.taux_refroidissement = taux_refroidissement
        self.bruit_min = bruit_min
        self.noise_coords = float(noise_coords)
        
        self.bead_atom = bead_atom
        self.best_score = float('inf')
        self.rsrnasp_weight = float(score_weight)

        # Paramètres FENE-Fraenkel
        self.k_fene = float(k)
        self.l0 = float(l0)

        # Constantes physiques de rsRNASP (selon l'article)
        self.step_distance = 0.3
        self.cutoff_short = 13.0
        self.cutoff_long = 24.0

        self.verbose = verbose

        self.load_dict_potentials()
        self.load_structure(sequence)
        self.setup_pairs()

    def load_dict_potentials(self):
        short_path = "potentials/short-ranged.potential"
        long_path = "potentials/long-ranged.potential"
        
        if os.path.exists(short_path) and os.path.exists(long_path):
            num_types, num_bins, dict_pots = load_rsrnasp_potentials(short_path, long_path)
            
            # Tenseur de forme : (2 états K, 85 types, 85 types, X bins de 0.3A)
            self.potential_tensor = torch.zeros((2, num_types, num_types, num_bins), dtype=torch.float32).to(self.device)
            
            for (k_state, t1, t2, dist_bin), energy in dict_pots.items():
                self.potential_tensor[k_state, t1, t2, dist_bin] = energy
            
            if self.verbose:
                print(f"✅ Potentiels rsRNASP chargés (Types: {num_types}, Bins: {num_bins}, Step: {self.step_distance}Å).")
        else:
            if self.verbose:
                print(f"⚠️ Fichiers de potentiels rsRNASP introuvables dans 'potentials/', rsRNASP ignoré.")
            self.potential_tensor = None

    def load_structure(self, sequence):
        self.num_beads = len(sequence)
        bead_coords = []
        bead_rows = []

        for i, nt in enumerate(sequence):
            # Initialisation en spirale pour casser la symétrie et aider les gradients
            phi = i * 0.5
            r = 2.0
            coord = [
                float(i * self.l0 * 0.1),
                float(r * np.cos(phi)),
                float(r * np.sin(phi))
            ]
            bead_coords.append(coord)
            
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
        
        if getattr(self, 'potential_tensor', None) is not None:
            mask_valid = sep > 0
            self.pair_i = i[mask_valid].to(self.device)
            self.pair_j = j[mask_valid].to(self.device)
            self.sep = sep[mask_valid].to(self.device)
            
            # Mapping K_state pour rsRNASP
            # Si |i-j| <= 4 -> k_state = 0 (Short range)
            # Si |i-j| >= 5 -> k_state = 1 (Long range)
            self.k_states = (self.sep >= 5).long()

            # Attribution des types 85
            t_vals = []
            for row in self.template_rows:
                res_name = str(row['residue_name'])
                atom_name = str(row['atom_name'])
                t = get_rsrnasp_type(res_name, atom_name)
                if t == -1: 
                    t = 0 # Type par défaut (AP) si non trouvé pour éviter les crashs GPU
                t_vals.append(t)
            
            t_vals_tensor = torch.tensor(t_vals, dtype=torch.long, device=self.device)
            self.t1_vals = t_vals_tensor[self.pair_i]
            self.t2_vals = t_vals_tensor[self.pair_j]

    def fene_fraenkel_bond_energy(self):
        p1 = self.coords[:-1]
        p2 = self.coords[1:]
        r = torch.norm(p2 - p1, dim=1) + 1e-8
        delta = r - self.l0
        return torch.sum(0.5 * self.k_fene * (delta ** 2))

    def rsrnasp_like_energy(self):
        if getattr(self, 'potential_tensor', None) is None or self.rsrnasp_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.pair_i]
        p_j = self.coords[self.pair_j]
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        # Mise à l'échelle de la distance par rapport à la taille des bins (0.3 Å)
        d_scaled = dists / self.step_distance
        
        max_idx = self.potential_tensor.size(3) - 1
        d_clamp = torch.clamp(d_scaled, 0.5, float(max_idx - 1.5)) 
        
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        im1 = torch.clamp(d0 - 1, min=0)
        i2 = torch.clamp(d1 + 1, max=max_idx)
        
        # Extraction depuis le tenseur avec les états K (0 pour short, 1 pour long)
        p0 = self.potential_tensor[self.k_states, self.t1_vals, self.t2_vals, im1]
        p1 = self.potential_tensor[self.k_states, self.t1_vals, self.t2_vals, d0]
        p2 = self.potential_tensor[self.k_states, self.t1_vals, self.t2_vals, d1]
        p3 = self.potential_tensor[self.k_states, self.t1_vals, self.t2_vals, i2]
        
        # Interpolation Catmull-Rom (Spline cubique)
        interp_energy = 0.5 * (
            (2 * p1) + (-p0 + p2) * alpha +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * alpha**2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * alpha**3
        )
        
        # Cutoffs dynamiques selon l'article : 13A pour short-range, 24A pour long-range
        cutoffs = torch.where(self.k_states == 0, 
                              torch.tensor(self.cutoff_short, device=self.device), 
                              torch.tensor(self.cutoff_long, device=self.device))
        
        # Cutoff sigmoid pour une annulation douce et continue
        cutoff_weights = torch.sigmoid(2.0 * (cutoffs - dists))
        
        score = torch.sum(interp_energy * cutoff_weights)
        
        return self.rsrnasp_weight * score

    def total_energy(self):
        bond = self.fene_fraenkel_bond_energy()
        rsrnasp = self.rsrnasp_like_energy()
        return bond + rsrnasp, bond, rsrnasp

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        best_coords = self.coords.detach().clone()
        self.best_score = float('inf')
        current_noise = self.noise_coords
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            print(f"🚀 Début de l'optimisation rsRNASP (Bassin Hopping/Recuit)")
            print(f"Device: {self.device} | Bruit initial: {current_noise}Å | LR: {self.lr}")

        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Phase d'Exploration {cycle_count} (Secousse: {current_noise:.4f}Å) ---")
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            while True:
                optimizer.zero_grad()
                total, bond, rsrnasp = self.total_energy()
                total.backward()
                
                # Clipping pour la stabilité physique
                torch.nn.utils.clip_grad_norm_([self.coords], max_norm=5.0)
                optimizer.step()

                current_loss = total.item()

                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0 
                
                prev_loss = current_loss
                epoch += 1

                if self.verbose and epoch % 100 == 0:
                    print(f"  Iter {epoch:4d} | Total: {current_loss:.4f} | FENE: {bond:.4f} | rsRNASP: {rsrnasp:.4f}")

                if patience_counter >= self.patience_locale:
                    if self.verbose: print(f"  🛑 Minimum local atteint en {epoch} itérations.")
                    break
                
                if epoch > 10000:
                    if self.verbose: print(f"  ⚠️ Coupure de sécurité à 10000 itérations.")
                    break

            if current_loss < (self.best_score - self.min_delta):
                if self.verbose: print(f"  🌟 Record absolu ! {self.best_score:.4f} -> {current_loss:.4f}")
                self.best_score = current_loss
                best_coords.copy_(self.coords.detach())
                cycles_sans_amelioration = 0
            else:
                cycles_sans_amelioration += 1
                if self.verbose: print(f"  ❌ Pas d'amélioration ({cycles_sans_amelioration}/{self.patience_globale}).")

            current_noise *= self.taux_refroidissement
            
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                with torch.no_grad():
                    self.coords.copy_(best_coords)
                    self.coords.add_(torch.randn_like(self.coords) * current_noise)
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        if self.verbose:
            print("\n✅ Optimisation globale terminée !")
            if current_noise <= self.bruit_min:
                print("👉 Système refroidi (bruit minimal atteint).")
            else:
                print("👉 Arrêt prématuré : stagnation du score global.")

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