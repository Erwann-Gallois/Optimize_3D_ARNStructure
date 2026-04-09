import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

# Importation du parseur spécifique pour rsRNASP
from parse_rsrnasp_potentials import load_rsrnasp_potentials, get_rsrnasp_type, ATOM_TYPES_85

class FullAtomRsRNASPOptimizer:
    def __init__(
        self, 
        pdb_path, 
        lr=0.2, 
        output_path="output_rigid.pdb", 
        ref_atom="C3'", 
        noise_coords=1.5, 
        noise_angles=0.5, 
        backbone_weight=100.0, 
        rsrnasp_weight=0.5, # Poids réduit pour éviter le molten globule
        clash_weight=100.0, # Poids fort pour interdire les chevauchements
        verbose=True,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01
    ):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"The PDB file {pdb_path} does not exist.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.noise_coords = float(noise_coords)
        self.noise_angles = float(noise_angles)
        self.patience_locale = patience_locale
        self.min_delta = min_delta
        self.patience_globale = patience_globale
        self.taux_refroidissement = taux_refroidissement
        self.bruit_min = bruit_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.best_score = float('inf')
        self.backbone_weight = backbone_weight
        self.rsrnasp_weight = rsrnasp_weight
        self.clash_weight = clash_weight
        
        # Paramètres physiques de rsRNASP
        self.step_distance = 0.3
        self.cutoff_short = 13.0
        self.cutoff_long = 24.0
        
        self.verbose = verbose
        if self.verbose:
            print(f"Using device: {self.device}")

        # 1. Chargement des potentiels
        self.load_dict_potentials()
        
        # 2. Chargement et préparation de la structure rigide
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

        # 3. Préparation des rayons de Van der Waals pour le Clash Stérique
        self.vdw_radii_map = self._get_vdw_radii()
        self.setup_clash_detection()

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
                print(f"rsRNASP potential files not found, rsRNASP ignored.")
            self.potential_tensor = None

    def convert_pdb_to_rigid_tensors(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        # Filtrage et typage rsRNASP
        df_atoms['rsrnasp_type'] = df_atoms.apply(lambda row: get_rsrnasp_type(row['residue_name'], row['atom_name']), axis=1)
        self.df_filtered = df_atoms[df_atoms['rsrnasp_type'] != -1].reset_index(drop=True)
        
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32).to(self.device)
        res_ids = self.df_filtered['residue_number'].values
        unique_res = np.unique(res_ids)
        
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in res_ids], dtype=torch.long).to(self.device)
        
        num_nucs = len(unique_res)
        ref_coords_init = torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device)
        
        for i, res in enumerate(unique_res):
            mask = (self.df_filtered['residue_number'] == res) & (self.df_filtered['atom_name'] == self.ref_atom)
            if mask.any():
                ref_idx = mask.idxmax()
                ref_coords_init[i] = raw_coords[ref_idx]
            else:
                first_idx = (self.df_filtered['residue_number'] == res).idxmax()
                ref_coords_init[i] = raw_coords[first_idx]

        # OPTIMIZABLE PARAMETERS (Rigid body)
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32).to(self.device))
        
        # CONSTANT: internal offsets
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # Identification Backbone
        self.bb_i_idx, self.bb_j_idx = [], [] 
        self.p_curr_idx, self.p_next_idx = [], [] 
        self.c3_idx = [] 

        for i in range(len(unique_res) - 1):
            res_curr, res_next = unique_res[i], unique_res[i+1]
            
            df_curr = self.df_filtered[self.df_filtered['residue_number'] == res_curr]
            df_next = self.df_filtered[self.df_filtered['residue_number'] == res_next]
            
            idx_o3 = df_curr[df_curr['atom_name'] == "O3'"].index
            idx_c3 = df_curr[df_curr['atom_name'] == "C3'"].index
            idx_p_c = df_curr[df_curr['atom_name'] == "P"].index
            idx_p_n = df_next[df_next['atom_name'] == "P"].index
            
            if not idx_o3.empty and not idx_p_n.empty:
                self.bb_i_idx.append(idx_o3[0])
                self.bb_j_idx.append(idx_p_n[0])
            if not idx_p_c.empty and not idx_p_n.empty:
                self.p_curr_idx.append(idx_p_c[0])
                self.p_next_idx.append(idx_p_n[0])
            if not idx_c3.empty and not idx_o3.empty and not idx_p_n.empty:
                self.c3_idx.append(idx_c3[0])
        
        self.bb_i_idx = torch.tensor(self.bb_i_idx, dtype=torch.long, device=self.device)
        self.bb_j_idx = torch.tensor(self.bb_j_idx, dtype=torch.long, device=self.device)
        self.p_curr_idx = torch.tensor(self.p_curr_idx, dtype=torch.long, device=self.device)
        self.p_next_idx = torch.tensor(self.p_next_idx, dtype=torch.long, device=self.device)
        self.c3_idx = torch.tensor(self.c3_idx, dtype=torch.long, device=self.device)
        
        # Préparation des paires rsRNASP
        atom_types = torch.tensor(self.df_filtered['rsrnasp_type'].values, dtype=torch.int32).to(self.device)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
        
        res_tensor = torch.tensor(res_ids, dtype=torch.int32).to(self.device)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        
        # rsRNASP ignore les atomes du même nucléotide (sep == 0)
        mask_inter = sep > 0
        
        self.pair_i = i_idx[mask_inter].to(torch.int32)
        self.pair_j = j_idx[mask_inter].to(torch.int32)
        self.sep = sep[mask_inter]
        
        # K_state: 0 pour court (1 à 4), 1 pour long (>= 5)
        self.k_states = (self.sep >= 5).long()
        self.t1_vals = atom_types[self.pair_i]
        self.t2_vals = atom_types[self.pair_j]

    def _get_vdw_radii(self):
        radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8}
        vdw_list = []
        # Le format des types rsRNASP (ex: "AP", "AOP1", "GN1")
        # L'élément atomique correspond presque toujours au deuxième caractère
        for t in ATOM_TYPES_85:
            element = t[1]
            vdw_list.append(radii.get(element, 1.7))
        return torch.tensor(vdw_list, dtype=torch.float32).to(self.device)

    def setup_clash_detection(self):
        r_i = self.vdw_radii_map[self.t1_vals.long()]
        r_j = self.vdw_radii_map[self.t2_vals.long()]
        # Distance minimale de VDW autorisée (tolérance de 15% de chevauchement)
        self.min_dist_threshold = (r_i + r_j) * 0.85

    def get_rotation_matrices(self):
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])

        N = self.rot_angles.shape[0]
        Rx = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a

        Ry = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = cos_b, sin_b, -sin_b, cos_b

        Rz = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = cos_g, -sin_g, sin_g, cos_g

        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_current_full_coords(self):
        R = self.get_rotation_matrices() 
        R_atoms = R[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def calculate_detailed_scores(self, current_full_coords):
        n_pairs = self.pair_i.size(0)
        chunk_size = 5000000 # Évite les Out Of Memory sur GPU
        
        rsrnasp_score = torch.tensor(0.0, device=self.device)
        max_idx = self.potential_tensor.size(3) - 1
        
        for start_idx in range(0, n_pairs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_pairs)
            
            p1 = current_full_coords[self.pair_i[start_idx:end_idx].long()]
            p2 = current_full_coords[self.pair_j[start_idx:end_idx].long()]
            
            dists = torch.norm(p1 - p2, dim=1) + 1e-8
            
            # Échelle par rapport à la taille des bins rsRNASP (0.3 Å)
            d_scaled = dists / self.step_distance
            d_clamp = torch.clamp(d_scaled, 0.5, float(max_idx - 1.5))
            
            d0 = torch.floor(d_clamp).long()
            d1 = d0 + 1
            alpha = d_clamp - d0.float()
            
            im1 = torch.clamp(d0 - 1, min=0)
            i2 = torch.clamp(d1 + 1, max=max_idx)
            
            k_st = self.k_states[start_idx:end_idx]
            t1 = self.t1_vals[start_idx:end_idx].long()
            t2 = self.t2_vals[start_idx:end_idx].long()
            
            p0 = self.potential_tensor[k_st, t1, t2, im1]
            p1_val = self.potential_tensor[k_st, t1, t2, d0]
            p2_val = self.potential_tensor[k_st, t1, t2, d1]
            p3 = self.potential_tensor[k_st, t1, t2, i2]
            
            # Interpolation Catmull-Rom (Spline Cubique)
            interp_energy = 0.5 * (
                (2 * p1_val) + (-p0 + p2_val) * alpha +
                (2 * p0 - 5 * p1_val + 4 * p2_val - p3) * alpha**2 +
                (-p0 + 3 * p1_val - 3 * p2_val + p3) * alpha**3
            )
            
            # Cutoffs dynamiques (13 Å ou 24 Å)
            cutoffs = torch.where(k_st == 0, 
                                  torch.tensor(self.cutoff_short, device=self.device), 
                                  torch.tensor(self.cutoff_long, device=self.device))
            
            cutoff_weights = torch.sigmoid(2.0 * (cutoffs - dists))
            rsrnasp_score = rsrnasp_score + torch.sum(interp_energy * cutoff_weights)

        # 1. Répulsion Stérique VDW
        p_i, p_j = current_full_coords[self.pair_i.long()], current_full_coords[self.pair_j.long()]
        dist_pairs = torch.norm(p_i - p_j, dim=1)

        thresholds = self.min_dist_threshold
        clash_penalty = torch.sum(torch.clamp(thresholds - dist_pairs, min=0.0)**2) * self.clash_weight

        # 2. Robustesse Backbone
        bb_penalty = torch.tensor(0.0, device=self.device)
        if len(self.bb_i_idx) > 0:
            p_o3, p_p = current_full_coords[self.bb_i_idx], current_full_coords[self.bb_j_idx]
            dist_o3p = torch.norm(p_o3 - p_p, dim=1)
            bb_penalty += torch.sum((dist_o3p - 1.61)**2) * self.backbone_weight

            if len(self.p_curr_idx) > 0:
                dist_pp = torch.norm(current_full_coords[self.p_curr_idx] - current_full_coords[self.p_next_idx], dim=1)
                bb_penalty += torch.sum((dist_pp - 5.9)**2) * (self.backbone_weight * 0.5)

            p_c3 = current_full_coords[self.c3_idx]
            v1, v2 = p_c3 - p_o3, p_p - p_o3
            cos_angle = torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1)*torch.norm(v2, dim=1) + 1e-8)
            bb_penalty += torch.sum((cos_angle - (-0.5))**2) * 20.0

        return rsrnasp_score * self.rsrnasp_weight, bb_penalty, clash_penalty

    def run_optimization(self):
        """
        Optimisation dynamique sans limite fixe d'époques ni de cycles.
        """
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()
        self.best_score = float('inf')

        if self.verbose:
            print(f"Starting dynamic rsRNASP optimization...")
            print(f"Number of atom pairs to process: {self.pair_i.size(0):,}")
        
        current_noise_c = self.noise_coords
        current_noise_a = self.noise_angles
        cycles_sans_amelioration = 0
        cycle_count = 0

        # EXTERNAL LOOP: Shakes (Exploration)
        while current_noise_c > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Exploration phase {cycle_count} (Noise: {current_noise_c:.4f}Å / {current_noise_a:.4f}rad) ---")
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            # INTERNAL LOOP: Local minimization
            while True:
                optimizer.zero_grad()
                
                coords = self.get_current_full_coords()
                score, bb_penalty, clash_penalty = self.calculate_detailed_scores(coords)
                loss = score + bb_penalty + clash_penalty

                loss.backward()
                
                # Gradient clipping pour la stabilité (essentiel avec les splines cubiques)
                torch.nn.utils.clip_grad_norm_([self.ref_coords, self.rot_angles], max_norm=5.0)
                optimizer.step()

                current_loss = loss.item()

                # Condition d'arrêt local (ΔE)
                delta_loss = abs(prev_loss - current_loss)
                if delta_loss < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                prev_loss = current_loss
                epoch += 1

                if (epoch % 1000 == 0) and self.verbose:
                    print(f"  Iteration {epoch:4d} | Total: {current_loss:.4f} | rsRNASP: {score.item():.4f} | BB: {bb_penalty.item():.2f} | Clash: {clash_penalty.item():.2f}")
                
                if patience_counter >= self.patience_locale:
                    if self.verbose:
                        print(f"Local minimum reached in {epoch} iterations.")
                    break
                
                if epoch > 10000:
                    if self.verbose:
                        print(f"Local phase too long, safety cutoff at 10000.")
                    break

                del coords, score, bb_penalty, clash_penalty, loss

            # --- CYCLE SUMMARY ---
            if current_loss < (self.best_score - self.min_delta):
                if self.verbose:
                    print(f"New absolute record! {self.best_score:.4f} -> {current_loss:.4f}")
                self.best_score = current_loss
                best_ref_coords.copy_(self.ref_coords)
                best_rot_angles.copy_(best_rot_angles)
                cycles_sans_amelioration = 0
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    print(f"No record. (Failures: {cycles_sans_amelioration}/{self.patience_globale})")

            # --- PRÉPARATION DU CYCLE SUIVANT ---
            current_noise_c *= self.taux_refroidissement
            current_noise_a *= self.taux_refroidissement
            
            if current_noise_c > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                if self.verbose:
                    print(f"Applying SHAKE. Returning to the best conformation and adding noise.")
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref_coords)
                    self.rot_angles.copy_(best_rot_angles)
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * current_noise_c)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * current_noise_a)
                
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
                torch.cuda.empty_cache()

        if self.verbose:
            print("\n Optimization finished.")
        
        with torch.no_grad():
            self.ref_coords.copy_(best_ref_coords)
            self.rot_angles.copy_(best_rot_angles)
            
        self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_full_coords = self.get_current_full_coords().detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_full_coords
        out_ppdb.df["ATOM"]["chain_id"] = "A"
        out_ppdb.to_pdb(path=self.output_path)
        
        if self.verbose:
            print(f"File saved: {self.output_path}")
            print(f"Best score: {self.best_score:.4f}")