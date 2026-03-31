import os
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type


class BeadSpringRASPOptimizer:
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
        type_RASP="c3",
        score_weight=1.0,
        verbose=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.output_path = output_path
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.noise_coords = float(noise_coords)
        self.bead_atom = bead_atom

        # Paramètres RASP
        self.type_RASP = type_RASP
        self.rasp_weight = float(score_weight)

        # Paramètres FENE-Fraenkel
        self.k = float(k)
        self.l0 = float(l0)

        self.verbose = verbose

        self.load_dict_potentials()
        self.load_structure(pdb_path)
        self.setup_pairs()

    def load_dict_potentials(self):
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            taille_mat, dict_pots = load_rasp_potentials(path)
            self.convert_dict_to_tensor(dict_pots, taille_mat)
            print(f"Potentiels RASP '{self.type_RASP}' chargés.")
        else:
            print(f"Fichier de potentiel non trouvé : {path}, RASP ignoré.")
            self.potential_tensor = None

    def convert_dict_to_tensor(self, dict_pots, taille_mat):
        self.potential_tensor = torch.zeros(taille_mat, dtype=torch.float32).to(self.device)
        for (k, t1, t2, dist), energy in dict_pots.items():
            if k < taille_mat[0] and t1 < taille_mat[1] and t2 < taille_mat[2] and dist < taille_mat[3]:
                self.potential_tensor[k, t1, t2, dist] = energy
                self.potential_tensor[k, t2, t1, dist] = energy

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
        
        # paires RASP
        if getattr(self, 'potential_tensor', None) is not None:
            mask_rasp = sep > 0
            self.rasp_i = i[mask_rasp].to(self.device)
            self.rasp_j = j[mask_rasp].to(self.device)
            self.rasp_sep = sep[mask_rasp].to(self.device)
            self.k_vals = torch.clamp(self.rasp_sep - 1, 0, 5)

            t_vals = []
            for row in self.template_rows:
                res_name = str(row['residue_name']).strip() if 'residue_name' in row else 'RNA'
                atom_name = str(row['atom_name']).strip() if 'atom_name' in row else self.bead_atom
                t = get_rasp_type(res_name, atom_name, self.type_RASP)
                if t == -1: t = 0
                t_vals.append(t)
            
            t_vals_tensor = torch.tensor(t_vals, dtype=torch.long, device=self.device)
            self.t1_vals = t_vals_tensor[self.rasp_i]
            self.t2_vals = t_vals_tensor[self.rasp_j]

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

    def rasp_like_energy(self):
        if getattr(self, 'potential_tensor', None) is None or self.rasp_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        p_i = self.coords[self.rasp_i]
        p_j = self.coords[self.rasp_j]
        
        dists = torch.norm(p_i - p_j, dim=1) + 1e-8
        
        max_idx = self.potential_tensor.size(3) - 1
        d_clamp = torch.clamp(dists, 0.5, float(max_idx - 1.5)) 
        d0 = torch.floor(d_clamp).long()
        d1 = d0 + 1
        alpha = d_clamp - d0.float()
        
        im1 = torch.clamp(d0 - 1, min=0)
        i2 = torch.clamp(d1 + 1, max=max_idx)
        
        p0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, im1]
        p1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        p2 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        p3 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, i2]
        
        # Interpolation Catmull-Rom
        interp_energy = 0.5 * (
            (2 * p1) + (-p0 + p2) * alpha +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * alpha**2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * alpha**3
        )
        
        # Cutoff sigmoid pour une annulation douce autour de 19A
        cutoff = torch.sigmoid(2.0 * (19.0 - dists))
        rasp_score = torch.sum(interp_energy * cutoff)
        
        return self.rasp_weight * rasp_score

    def total_energy(self):
        bond = self.fene_fraenkel_bond_energy()
        rasp = self.rasp_like_energy()
        return bond + rasp, bond, rasp

    def run_optimization(self):
        optimizer = torch.optim.Adam([self.coords], lr=self.lr)

        best_coords = self.coords.detach().clone()
        best_loss = float('inf')

        print(f"Utilisation du device : {self.device}")
        print(f"🚀 Début de l'optimisation bead-spring (Adam, {self.num_cycles} cycles)...")

        epochs_per_cycle = max(1, self.num_epochs // self.num_cycles)

        for cycle in range(self.num_cycles):
            print(f"\n--- Cycle {cycle+1}/{self.num_cycles} ---")
            
            for epoch in range(epochs_per_cycle):
                optimizer.zero_grad()

                total, bond, rasp = self.total_energy()
                total.backward()
                optimizer.step()

                if total.item() < best_loss:
                    best_loss = total.item()
                    best_coords.copy_(self.coords.detach())

                if epoch % 50 == 0 or epoch == epochs_per_cycle - 1:
                    print(
                        f"Epoch {epoch:4d} | Total: {total.item():.4f} | "
                        f"FENE-Fraenkel: {bond.item():.4f} | RASP-CG: {rasp.item():.4f}"
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
                # Nettoyage des données
                atom_name = str(row.get('atom_name', 'C3\'')).strip()
                res_name = str(row.get('residue_name', 'A')).strip()
                chain_id = str(row.get('chain_id', 'A')).strip()[:1] # 1 seul caractère
                res_num = int(row.get('residue_number', i))

                # Formatage PDB strict (Standard Protein Data Bank)
                # Colonne 13-16 : Nom de l'atome (aligné à gauche avec un espace avant si < 4 car.)
                # Colonne 18-20 : Nom du résidu (aligné à droite)
                # Colonne 22    : Chain ID
                # Colonne 23-26 : Numéro de résidu
                
                if len(atom_name) < 4:
                    atom_field = f" {atom_name:<3s}"
                else:
                    atom_field = f"{atom_name:<4s}"

                line = (
                    f"ATOM  {i:5d} {atom_field:4s} {res_name:>3s} {chain_id}{res_num:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           C\n"
                )
                f.write(line)
            f.write("END\n")

        print(f"Fichier sauvegardé et formaté pour Arena : {self.output_path}")