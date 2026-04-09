import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO, MMCIFIO, PDBParser
import subprocess

class BeadSpringRASPOptimizer:
    def __init__(
        self,
        sequence,
        lr=0.05,
        output_path="output_bead.pdb",
        noise_coords=3.0,
        bead_atom="C3'",
        k=45.86,        # From std=0.305
        l0=5.726,
        k_angle=30.0,   # Bending stiffness (adjusted for RASP)
        theta0=139.07,  # Calculated mean angle
        type_RASP="all",
        score_weight=50.0,
        verbose=True,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01,
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
        self.export_cif = export_cif
        self.verbose = verbose
        self.noise_coords = float(noise_coords)
        self.bead_atom = bead_atom
        self.best_score = float('inf')
        # RASP parameters
        self.type_RASP = type_RASP
        self.rasp_weight = float(score_weight)

        # FENE-Fraenkel parameters
        self.k = float(k)
        self.l0 = float(l0)
        self.k_angle = float(k_angle)
        self.theta0_rad = float(theta0) * np.pi / 180.0

        self.load_dict_potentials()
        self.load_structure(sequence)
        self.setup_pairs()

    def load_dict_potentials(self):
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            taille_mat, dict_pots = load_rasp_potentials(path)
            self.convert_dict_to_tensor(dict_pots, taille_mat)
            if self.verbose:
                print(f"RASP potentials '{self.type_RASP}' loaded.")
        else:
            if self.verbose:
                print(f"Potential file not found: {path}, RASP ignored.")
            self.potential_tensor = None

    def convert_dict_to_tensor(self, dict_pots, taille_mat):
        self.potential_tensor = torch.zeros(taille_mat, dtype=torch.float32).to(self.device)
        for (k, t1, t2, dist), energy in dict_pots.items():
            if k < taille_mat[0] and t1 < taille_mat[1] and t2 < taille_mat[2] and dist < taille_mat[3]:
                self.potential_tensor[k, t1, t2, dist] = energy
                self.potential_tensor[k, t2, t1, dist] = energy

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
        
        # Pairs for excluded volume (WCA) and RASP
        mask_rasp = sep > 1
        self.rasp_i = i[mask_rasp].to(self.device)
        self.rasp_j = j[mask_rasp].to(self.device)
        self.rasp_sep = sep[mask_rasp].to(self.device)

        if getattr(self, 'potential_tensor', None) is not None:
            self.k_vals = torch.clamp(self.rasp_sep - 1, 0, 5)

            t_vals = []
            for row in self.template_rows:
                res_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
                atom_name = str(row['atom_name']).strip() if 'atom_name' in row else self.bead_atom
                t = get_rasp_type(res_name, atom_name, self.type_RASP)
                if t == -1: t = 0 # Default type
                t_vals.append(t)
            
            t_vals_tensor = torch.tensor(t_vals, dtype=torch.long, device=self.device)
            self.t1_vals = t_vals_tensor[self.rasp_i]
            self.t2_vals = t_vals_tensor[self.rasp_j]

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

    def rasp_like_energy(self):
        if getattr(self, 'potential_tensor', None) is None or self.rasp_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.rasp_i]
        p_j = self.coords[self.rasp_j]
        
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        max_idx = self.potential_tensor.size(3) - 1
        d_clamp = torch.clamp(dists, 0.5, float(max_idx - 1.5)) # Clamp to grid range
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        im1 = torch.clamp(d0 - 1, min=0)
        i2 = torch.clamp(d1 + 1, max=max_idx)
        
        p0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, im1]
        p1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        p2 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        p3 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, i2]
        
        # Catmull-Rom interpolation
        interp_energy = 0.5 * (
            (2 * p1) + (-p0 + p2) * alpha +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * alpha**2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * alpha**3
        )
        
        # Sigmoid cutoff for smooth decay around 19A
        cutoff = torch.sigmoid(2.0 * (19.0 - dists))
        rasp_score = torch.sum(interp_energy * cutoff)
        
        return self.rasp_weight * rasp_score

    def excluded_volume_energy(self):
        if self.rasp_i is None or len(self.rasp_i) == 0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.rasp_i]
        p_j = self.coords[self.rasp_j]
    
        # Squared distance calculation for performance
        dist_sq = torch.sum((p_i - p_j)**2, dim=1) + 1e-8
        
        # WCA parameters
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
        
        # WCA formula: 4 * epsilon * (r12 - r6) + epsilon
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
        rasp = self.rasp_like_energy()
        angle = self.valence_angle_energy()
        repulsion = self.excluded_volume_energy()
        return bond + rasp + angle + repulsion, bond, rasp, angle, repulsion

    def run_optimization(self):
        """
        Dynamic optimization without fixed limit of epochs or cycles.
        
        - patience_locale : Number of iterations without improvement before finishing a folding phase.
        - patience_globale : Number of consecutive "shakes" without beating the absolute record before giving up.
        - taux_refroidissement : Noise reduction factor at each shake (e.g., 0.85).
        - bruit_min : Threshold below which the shake is considered negligible.
        """
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        best_coords = self.coords.detach().clone()
        self.best_score = float('inf')
        
        # Initial noise replaces the "number of cycles" concept
        current_noise = self.noise_coords
        cycles_sans_amelioration = 0
        cycle_count = 0

        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Starting dynamic optimization (Basin Hopping/Annealing Algorithm)...")

        # EXTERNAL LOOP: Controlled by noise level and successive failures
        while current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Exploration Phase {cycle_count} (Shake noise: {current_noise:.4f}Å) ---")
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            # INTERNAL LOOP: Replaces 'for epoch in range(num_epochs)'
            while True:
                optimizer.zero_grad()
                total, bond, rasp, angle, repulsion = self.total_energy()
                total.backward()
                optimizer.step()

                current_loss = total.item()

                if current_loss < self.best_score:
                    self.best_score = current_loss
                    best_coords.copy_(self.coords.detach())
                # Local stop condition (ΔE)
                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0  # Found a good slope, reset
                
                prev_loss = current_loss
                epoch += 1

                # Display every 100 steps to monitor
                if epoch % 1000 == 0:
                    if self.verbose:
                        print(f"  Iteration {epoch:4d} | Total: {current_loss:.4f} | FENE: {bond:.4f} | RASP: {rasp:.4f} | Angle: {angle:.4f} | Repulsion: {repulsion:.4f}")

                # Local phase stop if stuck in a minimum
                if patience_counter >= self.patience_locale:
                    if self.verbose:
                        print(f"  Local minimum reached in {epoch} iterations.")
                    break
                
                # Safety: relief valve to avoid a pure infinite loop (e.g., oscillation)
                if epoch > 10000:
                    if self.verbose:
                        print(f"  Local phase too long, safety cutoff at 10000.")
                    break

            # --- CYCLE SUMMARY ---
            # Is this local minimum the best ever found?
            if current_loss < (self.best_score - self.min_delta):
                if self.verbose:
                    print(f"  New absolute record! {self.best_score:.4f} -> {current_loss:.4f}")
                self.best_score = current_loss
                best_coords.copy_(self.coords.detach())
                cycles_sans_amelioration = 0  # Reset failure counter to zero
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    print(f"  No record. (Unsuccessful attempts : {cycles_sans_amelioration}/{self.patience_globale})")

            # --- NEXT CYCLE PREPARATION ---
            current_noise *= self.taux_refroidissement  # The system "cools"
            
            if current_noise > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                if self.verbose:
                    print(f"  -> Applying SHAKE. Returning to the best conformation and adding noise.")
                with torch.no_grad():
                    self.coords.copy_(best_coords)
                    self.coords.add_(torch.randn_like(self.coords) * current_noise)
                # Reset the optimizer to clear the "memory" (momentum) of previous gradients
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        # --- END OF OPTIMIZATION ---
        if self.verbose:
            print("\n Global optimization finished!")
            if current_noise <= self.bruit_min:
                print("Reason: The system has cooled (the noise has become too weak to break the bonds).")
            else:
                print(f"Reason: Inability to find a better folding after {self.patience_globale} consecutive shakes.")

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
            # Extract metadata
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

        # 3. Write file with PDBIO
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.output_path)
        # Split each space of your command into a list element
        commande = [
            "./../Arena/Arena", 
            self.output_path, 
            self.output_path.replace(".pdb", "_full_atom.pdb"), 
            "5"
        ]

        try:
            # Launch execution
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