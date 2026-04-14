import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO, MMCIFIO, PDBParser
import subprocess
import click
from tqdm import tqdm
class BeadSpringDFIREOptimizer:
    def __init__(
        self,
        sequence,
        lr=0.05,
        output_path="output_bead.pdb",
        noise_coords=1.5,
        bead_atom="C3'",
        k=20.0,
        l0=5.5,
        type_RASP="all",
        score_weight=50.0,
        verbose=True,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01,
        k_angle=30.0,   # Bending stiffness (adjusted for RASP)
        theta0=139.07,  # Calculated mean angle
        export_cif=False
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

        # Angle parameters
        self.k_angle = float(k_angle)
        self.theta0_rad = float(theta0) * np.pi / 180.0


        # RASP parameters
        self.type_RASP = type_RASP
        self.dfire_weight = float(score_weight)

        # FENE-Fraenkel parameters
        self.k = float(k)
        self.l0 = float(l0)
        self.export_cif = export_cif

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
                click.secho(f"DFIRE potentials loaded successfully from {path}.", fg='green')
        else:
            if self.verbose:
                click.secho(f"Potential file not found: {path}, DFIRE ignored.", fg='red')
                return 0
            self.potential_tensor = None

    def convert_dict_to_tensor(self, dict_pots):
        # Get all unique atom types
        all_types = set()
        for t1, t2 in dict_pots.keys():
            all_types.add(t1)
            all_types.add(t2)
        
        self.sorted_types = sorted(list(all_types))
        self.type_to_idx = {t: i for i, t in enumerate(self.sorted_types)}
        num_types = len(self.sorted_types)
        num_bins = len(next(iter(dict_pots.values())))
        
        # Tensor creation (num_types, num_types, num_bins)
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
            bead_coords.append([i * self.l0, 0.0, 0.0])  # Initial linear conformation
            # Store minimal necessary info for setup_pairs and save_pdb
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
        
        # DFIRE pairs
        if getattr(self, 'potential_tensor', None) is not None:
            mask_dfire = sep > 1  # Sequence separation > 1 for CG DFIRE
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
                    t_vals.append(0) # Default type if unknown
            
            t_vals_tensor = torch.tensor(t_vals, dtype=torch.long, device=self.device)
            self.t1_vals = t_vals_tensor[self.dfire_i]
            self.t2_vals = t_vals_tensor[self.dfire_j]

    def fene_fraenkel_bond_energy(self):
        """
        FENE-Fraenkel potential on consecutive bonds.

        We set: delta = r - l0
        E = 1/2 * k * (delta)^2
        """
        p1 = self.coords[:-1]
        p2 = self.coords[1:]

        r = torch.norm(p2 - p1, dim=1) + 1e-8
        delta = r - self.l0

        energy = 0.5 * self.k * (delta ** 2)
        return torch.sum(energy)

    def excluded_volume_energy(self):
        if self.dfire_i is None or len(self.dfire_i) == 0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.dfire_i]
        p_j = self.coords[self.dfire_j]
    
        # Squared distance calculation for performance
        dist_sq = torch.sum((p_i - p_j)**2, dim=1) + 1e-8
        
        # WCA parameters
        # sigma: bead diameter. 4.0A to 5.0A is reasonable for a residue.
        sigma = 4.5 
        epsilon = 1.0 # Barrier strength
        
        # r_cut = 2^(1/6) * sigma
        r_cut_sq = (2**(1/3)) * (sigma**2)
        
        # Mask to keep only colliding beads
        mask = dist_sq < r_cut_sq
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # Only calculate for close pairs
        d2 = dist_sq[mask]
        s2 = sigma**2
        
        # (sigma^2 / r^2)^3 = (sigma^6 / r^6)
        inv_r6 = (s2 / d2)**3
        inv_r12 = inv_r6**2
        
        # Formule WCA : 4 * epsilon * (r12 - r6) + epsilon
        wca = 4 * epsilon * (inv_r12 - inv_r6) + epsilon
        
        return 0.5 * torch.sum(wca)

    def dfire_like_energy(self):
        if getattr(self, 'potential_tensor', None) is None or self.dfire_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.dfire_i]
        p_j = self.coords[self.dfire_j]
        
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        step = 0.7
        max_dist = 19.6
        num_bins = self.potential_tensor.size(2)
        
        # Interpolation Catmull-Rom
        d_scaled = dists / step
        d_clamp = torch.clamp(d_scaled, 0.0, float(num_bins - 1.0001))
        
        i = torch.floor(d_clamp).long()
        t = d_clamp - i.float()
        
        # Gestion des bords pour les points de contrôle
        idx0 = torch.clamp(i - 1, min=0)
        idx1 = i
        idx2 = torch.clamp(i + 1, max=num_bins - 1)
        idx3 = torch.clamp(i + 2, max=num_bins - 1)
        
        p0 = self.potential_tensor[self.t1_vals, self.t2_vals, idx0]
        p1 = self.potential_tensor[self.t1_vals, self.t2_vals, idx1]
        p2 = self.potential_tensor[self.t1_vals, self.t2_vals, idx2]
        p3 = self.potential_tensor[self.t1_vals, self.t2_vals, idx3]
        
        t2 = t * t
        t3 = t2 * t
        
        f0 = -0.5 * t3 + t2 - 0.5 * t
        f1 = 1.5 * t3 - 2.5 * t2 + 1.0
        f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        f3 = 0.5 * t3 - 0.5 * t2
        
        interp_energy = f0 * p0 + f1 * p1 + f2 * p2 + f3 * p3
        
        # Sigmoid cutoff pour une décroissance douce autour de 19.6A
        cutoff = torch.sigmoid(2.0 * (max_dist - dists))
        dfire_score = torch.sum(interp_energy * cutoff)
        
        return self.dfire_weight * dfire_score

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
        dfire = self.dfire_like_energy()
        # repulsion = self.excluded_volume_energy()
        # angle = self.valence_angle_energy()
        return bond + dfire, bond, dfire

    def run_optimization(self):
        """
        Optimisation par Basin Hopping / Simulated Annealing avec barres tqdm.
        """
        # Initialisation
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)
        best_coords = self.coords.detach().clone()
        self.best_score = float('inf')
        
        current_noise = self.noise_coords
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            click.secho(f'🚀 Using device: {self.device}', fg='cyan', bold=True)
            click.secho("Starting dynamic optimization...", fg='cyan')

        # --- BARRE GLOBALE (Cycles de Shake) ---
        # On ne connaît pas le total, donc on laisse vide.
        shake_pbar = tqdm(desc="Global Optimization", unit=" cycles", dynamic_ncols=True)

        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            
            phase_best_score = float('inf')
            phase_best_coords = self.coords.detach().clone()
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0

            # --- BARRE LOCALE (Itérations Adam) ---
            # leave=False permet de l'effacer quand la phase est finie
            pbar = tqdm(total=10000, desc=f"  └─ Phase {cycle_count}", unit="it", leave=False, dynamic_ncols=True)
            
            while True:
                optimizer.zero_grad()
                total, bond, dfire = self.total_energy()
                total.backward()
                optimizer.step()

                current_loss = total.item()

                if current_loss < phase_best_score:
                    phase_best_score = current_loss
                    phase_best_coords.copy_(self.coords.detach())

                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                prev_loss = current_loss
                epoch += 1

                # Mise à jour de la barre locale
                pbar.update(1)
                if epoch % 100 == 0:
                    pbar.set_postfix({
                        "Total": f"{current_loss:.2f}",
                        "Bond": f"{bond.item():.2f}",
                        "DFIRE": f"{dfire.item():.2f}"
                    })                                                  

                # Conditions d'arrêt
                if patience_counter >= self.patience_locale or epoch >= 10000:
                    break
            
            pbar.close() # On ferme la barre locale proprement ici (une seule fois)

            # --- BILAN DE LA PHASE ---
            if phase_best_score < (self.best_score - self.min_delta):
                if self.verbose:
                    # Utiliser tqdm.write pour ne pas casser les barres
                    shake_pbar.write(click.style(f"🌟 New record: {phase_best_score:.4f}", fg='green', bold=True))
                
                self.best_score = phase_best_score
                best_coords.copy_(phase_best_coords)
                cycles_sans_amelioration = 0
            else:
                cycles_sans_amelioration += 1

            # Mise à jour de la barre globale (UNE SEULE FOIS par cycle de shake)
            shake_pbar.update(1)
            shake_pbar.set_postfix({
                "Best": f"{self.best_score:.2f}", 
                "Fail": f"{cycles_sans_amelioration}/{self.patience_globale}",
                "Noise": f"{current_noise:.3f}"
            })

            # --- PRÉPARATION DU SHAKE ---
            current_noise *= self.taux_refroidissement
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                with torch.no_grad():
                    self.coords.copy_(best_coords)
                    self.coords.add_(torch.randn_like(self.coords) * current_noise)
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        shake_pbar.close()

        # --- FIN ---
        if self.verbose:
            click.secho("\nGlobal optimization finished!", fg='green', bold=True)
            if current_noise <= self.bruit_min:
                click.secho("Reason: Minimum noise reached.", fg="yellow")
            else:
                click.secho(f"Reason: Max patience reached ({self.patience_globale} fails).", fg="yellow")

        with torch.no_grad():
            self.coords.copy_(best_coords)
        self.save_pdb()
    def save_pdb(self):
        # 1. Retrieve optimized coordinates
        coords = self.coords.detach().cpu().numpy()

        # 2. Create Biopython hierarchical structure
        structure = Structure.Structure("optimized")
        model = Model.Model(0)
        structure.add(model)

        # Dictionary to store created chains
        chains = {}

        for i, (row, coord) in enumerate(zip(self.template_rows, coords), start=1):
            # Proper metadata extraction
            chain_id = str(row['chain_id']).strip() if 'chain_id' in row and str(row['chain_id']).strip() else 'A'
            res_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
            res_num = int(row['residue_number']) if 'residue_number' in row else i
            atom_name = str(row['atom_name']).strip() if 'atom_name' in row else self.bead_atom
            
            # Chain management
            if chain_id not in chains:
                new_chain = Chain.Chain(chain_id)
                model.add(new_chain)
                chains[chain_id] = new_chain
            
            current_chain = chains[chain_id]

            # Residue creation (id = (' ', number, ' '))
            # Note: Biopython uses a tuple for the residue ID (hetero-flag, sequence_number, insertion_code)
            residue = Residue.Residue((' ', res_num, ' '), res_name, ' ')
            
            # Atom creation
            # Atom.Atom(name, coord, b_factor, occupancy, altloc, fullname, serial_number, element)
            atom = Atom.Atom(
                atom_name, 
                coord.tolist(), 
                0.0,    # B-factor
                1.0,    # Occupancy
                ' ',    # Altloc
                f" {atom_name:<3}", # Fullname (4 characters with spaces)
                i,      # Serial number
                element='C' # Element
            )
            
            residue.add(atom)
            current_chain.add(residue)

        # 3. Writing the file with PDBIO
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.output_path)
        arena_executable = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Arena", "Arena")
        )
        if not os.path.isfile(arena_executable):
            if self.verbose:
                click.secho(f"Arena executable not found: {arena_executable}", fg='red')
            return

        # Split each space of your command into a list element
        commande = [
            arena_executable,
            self.output_path, 
            self.output_path.replace(".pdb", "_full_atom.pdb"), 
            "5"
        ]

        try:
            # Display Arena output
            result = subprocess.run(commande, capture_output=True, text=True, check=True)
            if self.verbose:
                click.secho(f"Well-formatted PDB file saved: {self.output_path.replace('.pdb', '_full_atom.pdb')}", fg='green')
                click.secho(f"Best score: {self.best_score}", fg='green')
            # os.remove(self.output_path)

        except subprocess.CalledProcessError as e:
            if self.verbose:
                click.secho("Error during Arena execution:", fg='red')
                click.secho(e.stderr, fg='red')

        if self.export_cif:
            self.save_as_cif(self.output_path.replace(".pdb", "_full_atom.pdb"))

    def save_as_cif(self, pdb_path):
        """Converts a PDB file to CIF format using BioPython."""
        if not os.path.exists(pdb_path):
            if self.verbose:
                click.secho(f"Error: File {pdb_path} not found for CIF export.", fg='red')
            return
        
        cif_path = pdb_path.replace(".pdb", ".cif")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("result", pdb_path)
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_path)
        if self.verbose:
            click.secho(f"Structure also saved in CIF format: {cif_path}", fg='green')
