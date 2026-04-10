import os
import torch
import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO, MMCIFIO, PDBParser
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
        score_weight=50.0,
        verbose=True,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01,
        k_angle=30.0,   # Bending stiffness (adjusted for RASP)
        theta0=139.07,  # Calculated mean angle
        export_cif=False,
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

        # Paramètres de la force de rappel (Hooke)
        self.k_angle = float(k_angle)
        self.theta0_rad = float(theta0) * np.pi / 180.0

        # Paramètres FENE-Fraenkel
        self.k_fene = float(k)
        self.l0 = float(l0)
        self.export_cif = export_cif

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
                print(f"rsRNASP potentials loaded (Types: {num_types}, Bins: {num_bins}, Step: {self.step_distance}Å).")
        else:
            if self.verbose:
                print(f"rsRNASP potential files not found in 'potentials/', rsRNASP ignored.")
            self.potential_tensor = None

    def load_structure(self, sequence):
        self.num_beads = len(sequence)
        bead_coords = []
        bead_rows = []

        for i, nt in enumerate(sequence):
            if i == 0:
                current_pos = torch.zeros(3)
            else:
                direction = torch.randn(3)
                direction /= (torch.norm(direction) + 1e-8)
                current_pos = torch.tensor(bead_coords[-1]) + direction * self.l0
            bead_coords.append(current_pos.tolist())
            
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
        
        # Paires pour volume exclu (WCA) et RsRNASP
        mask_valid = sep > 1
        self.pair_i = i[mask_valid].to(self.device)
        self.pair_j = j[mask_valid].to(self.device)
        self.sep = sep[mask_valid].to(self.device)

        if getattr(self, 'potential_tensor', None) is not None:
            
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

    def excluded_volume_energy(self):
        if self.pair_i is None or len(self.pair_i) == 0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.pair_i]
        p_j = self.coords[self.pair_j]
    
        # Calcul des distances au carré pour la performance
        dist_sq = torch.sum((p_i - p_j)**2, dim=1) + 1e-8
        
        # Paramètres WCA
        sigma = 4.5 
        epsilon = 1.0 # Force de la barrière
        
        # r_cut = 2^(1/6) * sigma
        r_cut_sq = (2**(1/3)) * (sigma**2)
        
        # Masque pour ne garder que les perles en collision
        mask = dist_sq < r_cut_sq
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # On ne calcule que pour les paires proches
        d2 = dist_sq[mask]
        s2 = sigma**2
        
        # (sigma^2 / r^2)^3 = (sigma^6 / r^6)
        inv_r6 = (s2 / d2)**3
        inv_r12 = inv_r6**2
        
        # Formule WCA : 4 * epsilon * (r12 - r6) + epsilon
        wca = 4 * epsilon * (inv_r12 - inv_r6) + epsilon
        
        return 0.5 * torch.sum(wca)

    def valence_angle_energy(self):
        """Bending stiffness to avoid crushed 'spring' effect."""
        if self.num_beads < 3: return torch.tensor(0.0, device=self.device)
        v1 = self.coords[:-2] - self.coords[1:-1]
        v2 = self.coords[2:] - self.coords[1:-1]
        dot = torch.sum(v1 * v2, dim=1)
        norms = torch.norm(v1, dim=1) * torch.norm(v2, dim=1)
        cos_theta = torch.clamp(dot / (norms + 1e-8), -0.999, 0.999)
        theta = torch.acos(cos_theta)
        energy = 0.5 * self.k_angle * (theta - self.theta0_rad)**2
        return torch.sum(energy)

    def total_energy(self):
        bond = self.fene_fraenkel_bond_energy()
        rsrnasp = self.rsrnasp_like_energy()
        repulsion = self.excluded_volume_energy()
        angle = self.valence_angle_energy()
        return bond + rsrnasp + repulsion + angle, bond, rsrnasp, repulsion, angle

    def run_optimization(self):
        """
        Optimisation par Basin Hopping / Simulated Annealing.
        Capture le meilleur minimum local à chaque phase et le compare au record global.
        """
        # Initialisation de l'optimiseur sur les coordonnées actuelles
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        # Stockage du record absolu (global)
        best_coords = self.coords.detach().clone()
        self.best_score = float('inf')
        
        # Gestion du bruit et des cycles
        current_noise = self.noise_coords
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Starting dynamic optimization (Basin Hopping/Annealing Algorithm)...")

        # --- BOUCLE EXTERNE : Exploration des bassins énergétique (Global) ---
        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Exploration Phase {cycle_count} (Shake noise: {current_noise:.4f}Å) ---")
            
            # Variables pour capturer le meilleur moment de CETTE phase locale
            phase_best_score = float('inf')
            phase_best_coords = self.coords.detach().clone()
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            # --- BOUCLE INTERNE : Descente de gradient locale (Adam) ---
            while True:
                optimizer.zero_grad()
                total, bond, rsrnasp, repulsion, angle = self.total_energy()
                total.backward()
                optimizer.step()

                current_loss = total.item()

                # Sauvegarde du meilleur point rencontré durant la descente actuelle
                if current_loss < phase_best_score:
                    phase_best_score = current_loss
                    phase_best_coords.copy_(self.coords.detach())

                # Condition d'arrêt local basée sur la variation d'énergie (ΔE)
                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0  # On progresse encore, on réinitialise la patience
                
                prev_loss = current_loss
                epoch += 1

                # Affichage de suivi toutes les 1000 itérations
                if epoch % 1000 == 0 and self.verbose:
                    print(f"  Iteration {epoch:4d} | Total: {current_loss:.4f} | FENE: {bond:.4f} | "
                          f"rsRNASP: {rsrnasp:.4f} | Repulsion: {repulsion:.4f} | Angle: {angle:.4f}")

                # Arrêt si le minimum local est stabilisé
                if patience_counter >= self.patience_locale:
                    if self.verbose:
                        print(f"Local minimum reached in {epoch} iterations.")
                    break
                
                # Valve de sécurité pour éviter les boucles infinies ou oscillations
                if epoch >= 10000:
                    if self.verbose:
                        print(f"Local phase too long, safety cutoff at 10000.")
                    break

            # --- BILAN DE LA PHASE : Comparaison avec le Record Global ---
            # On compare le meilleur score de cette phase avec le record absolu de l'instance
            if phase_best_score < (self.best_score - self.min_delta):
                if self.verbose:
                    print(f"New absolute record! {self.best_score:.4f} -> {phase_best_score:.4f}")
                
                self.best_score = phase_best_score
                best_coords.copy_(phase_best_coords)
                cycles_sans_amelioration = 0  # On a trouvé un nouveau bassin, on réinitialise les échecs
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    print(f"No record. Phase best: {phase_best_score:.2f} | Global best: {self.best_score:.2f} "
                          f"({cycles_sans_amelioration}/{self.patience_globale})")

            # --- PRÉPARATION DU CYCLE SUIVANT (SHAKE) ---
            current_noise *= self.taux_refroidissement  # Le système "refroidit"
            
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                if self.verbose:
                    print(f"Applying SHAKE. Returning to the best conformation and adding noise.")
                
                with torch.no_grad():
                    # On repart toujours de la meilleure structure connue
                    self.coords.copy_(best_coords)
                    # On applique une perturbation aléatoire
                    self.coords.add_(torch.randn_like(self.coords) * current_noise)
                
                # Réinitialisation de l'optimiseur pour oublier les moments/gradients de la phase précédente
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        # --- FIN DE L'OPTIMISATION ---
        if self.verbose:
            print("\nGlobal optimization finished!")
            if current_noise <= self.bruit_min:
                print("Reason: Noise level reached the minimum threshold (bruit_min).")
            else:
                print(f"Reason: Failed to improve after {self.patience_globale} consecutive attempts.")

        # On charge la meilleure structure finale avant la sauvegarde
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
        arena_executable = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Arena", "Arena")
        )
        if not os.path.isfile(arena_executable):
            if self.verbose:
                print(f"Arena executable not found: {arena_executable}")
            return

        # On sépare chaque espace de votre commande en un élément de liste
        commande = [
            arena_executable,
            self.output_path, 
            self.output_path.replace(".pdb", "_full_atom.pdb"), 
            "5"
        ]

        try:
            # Lancement de l'exécution
            resultat = subprocess.run(commande, capture_output=True, text=True, check=True)
            
            # Display Arena output
            if self.verbose:
                print(f"Well-formatted PDB file saved: {self.output_path.replace('.pdb', '_full_atom.pdb')}")
                print(f"Best score: {self.best_score}")
            # os.remove(self.output_path)

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print("Error during Arena execution:")
                print(e.stderr)

        if self.export_cif:
            self.save_as_cif(self.output_path.replace(".pdb", "_full_atom.pdb"))

    def save_as_cif(self, pdb_path):
        """Converts a PDB file to CIF format using BioPython."""
        if not os.path.exists(pdb_path):
            if self.verbose:
                print(f"Error: File {pdb_path} not found for CIF export.")
            return
        
        cif_path = pdb_path.replace(".pdb", ".cif")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("result", pdb_path)
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_path)
        if self.verbose:
            print(f"Structure also saved in CIF format: {cif_path}")