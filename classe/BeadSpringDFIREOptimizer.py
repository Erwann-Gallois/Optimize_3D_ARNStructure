import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type


class BeadSpringDFIREOptimizer:
    def __init__(
        self,
        pdb_path,
        lr=0.05,
        output_path="output_bead.pdb",
        num_epochs=500,
        num_cycles=5,
        noise_coords=1.5,
        bead_atom="C3'",
        k=40.0,
        l0=5.5,
        exclude_near_neighbors=2,
        score_weight=1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.output_path = output_path
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.noise_coords = float(noise_coords)
        self.bead_atom = bead_atom

        # Paramètres DFIRE
        self.dfire_weight = float(score_weight)

        # Paramètres FENE-Fraenkel
        self.k = float(k)
        self.l0 = float(l0)

        self.load_dict_potentials()
        self.load_structure(pdb_path)
        self.setup_pairs()

    def load_dict_potentials(self):
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            self.convert_dict_to_tensor(dict_pots)
            print(f"Potentiels DFIRE chargés.")
        else:
            print(f"Fichier de potentiel non trouvé : {path}, DFIRE ignoré.")
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

    def load_structure(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df = ppdb.df['ATOM'].copy()

        unique_res = np.unique(df['residue_number'].values)
        bead_coords = []
        bead_rows = []

        for res_id in unique_res:
            group = df[df['residue_number'] == res_id]

            # On choisit un atome de référence si possible (plus stable qu'un centre de masse brut)
            mask = group['atom_name'] == self.bead_atom
            if mask.any():
                row = group[mask].iloc[0]
                coord = row[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)
                bead_rows.append(row)
            else:
                xyz = group[['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32)
                coord = np.mean(xyz, axis=0)
                bead_rows.append(group.iloc[0])

            bead_coords.append(coord)

        bead_coords = np.asarray(bead_coords, dtype=np.float32)
        self.coords = torch.nn.Parameter(torch.tensor(bead_coords, dtype=torch.float32, device=self.device))
        self.template_rows = bead_rows
        self.num_beads = self.coords.shape[0]

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
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        best_coords = self.coords.detach().clone()
        best_loss = float('inf')

        print(f"Utilisation du device : {self.device}")
        print(f"🚀 Début de l'optimisation bead-spring (Adam, {self.num_cycles} cycles)...")

        for cycle in range(self.num_cycles):
            print(f"\n--- Cycle {cycle+1}/{self.num_cycles} ---")
            
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()

                total, bond, dfire = self.total_energy()
                total.backward()
                optimizer.step()

                if total.item() < best_loss:
                    best_loss = total.item()
                    best_coords.copy_(self.coords.detach())

                if epoch % 50 == 0 or epoch == self.num_epochs - 1:
                    print(
                        f"Epoch {epoch:4d} | Total: {total.item():.4f} | "
                        f"FENE-Fraenkel: {bond.item():.4f} | "
                        f"DFIRE-CG: {dfire.item():.4f}"
                    )

            # SHAKE
            if cycle < self.num_cycles - 1:
                decay = 1.0 - (cycle / (self.num_cycles - 1))
                current_noise_c = self.noise_coords * decay
                
                print(f"Secousse aléatoire ! (Bruit: {current_noise_c:.5f}Å)")
                
                with torch.no_grad():
                    # Repartir du meilleur état pour pas divergé
                    self.coords.copy_(best_coords)
                    # Bruit gaussien
                    self.coords.add_(torch.randn_like(self.coords) * current_noise_c)

                # Reset de l'optimiseur pour annuler le momentum de Adam
                optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        print("\nOptimisation terminée, on restaure la meilleure conformation trouvée...")
        with torch.no_grad():
            self.coords.copy_(best_coords)

        self.save_pdb()

    def save_pdb(self):
        coords = self.coords.detach().cpu().numpy()

        with open(self.output_path, 'w') as f:
            for i, (row, c) in enumerate(zip(self.template_rows, coords), start=1):
                atom_name = str(row['atom_name']).strip() if 'atom_name' in row else 'P'
                residue_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
                chain_id = str(row['chain_id']).strip() if 'chain_id' in row and str(row['chain_id']).strip() else 'A'
                residue_number = int(row['residue_number']) if 'residue_number' in row else i

                line = (
                    f"ATOM  {i:5d} {atom_name:^4s} {residue_name:>3s} {chain_id}{residue_number:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n"
                )
                f.write(line)
            f.write("END\n")

        print(f"Fichier sauvegardé : {self.output_path}")
