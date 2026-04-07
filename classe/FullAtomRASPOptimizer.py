import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class FullAtomRASPOptimizer:
    def __init__(
        self, 
        pdb_path, 
        lr=0.2, 
        type_RASP="all", 
        verbose=False, 
        output_path="output_rigid.pdb", 
        ref_atom="C3'", 
        noise_coords=1.5, 
        noise_angles=0.5, 
        backbone_weight=100.0,
        patience_locale=100, 
        min_delta=1e-4, 
        patience_globale=5, 
        taux_refroidissement=0.85, 
        bruit_min=0.01
    ):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.type_RASP = type_RASP
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.noise_coords = float(noise_coords)
        self.noise_angles = float(noise_angles)
        self.patience_locale = patience_locale
        self.min_delta = min_delta
        self.patience_globale = patience_globale
        self.taux_refroidissement = taux_refroidissement
        self.bruit_min = bruit_min
        self.verbose = verbose
        self.clash_weight = 100.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Utilisation du device : {self.device}")
        self.best_score = float('inf')
        self.backbone_weight = backbone_weight
        # 1. Chargement des potentiels
        self.load_dict_potentials()
        
        # 2. Chargement et préparation de la structure rigide
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

        # 3. Détection des clashes
        self.vdw_radii_map = self._get_vdw_radii()
        self.setup_clash_detection()

    def load_dict_potentials(self):
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            taille_mat, dict_pots = load_rasp_potentials(path)
            self.convert_dict_to_tensor(dict_pots, taille_mat)
        else:
            print(f"Fichier de potentiel non trouvé : {path}")

    def convert_dict_to_tensor(self, dict_pots, taille_mat):
        # Pour RASP "all", on a 23 types (0-22). 
        # On définit explicitement les éléments pour que _get_vdw_radii fonctionne.
        if self.type_RASP == "all":
            # Mapping des 23 types RASP vers leurs éléments chimiques respectifs
            self.sorted_types = ["O", "P", "O", "C", "C", "O", "C", "O", "N", "C", "N", "C", "C", "C", "C", "N", "C", "C", "O", "C", "C", "C", "N"]
        elif self.type_RASP == "c3":
            self.sorted_types = ["A", "C", "G", "U"] # Fallback for residue-based
        else:
            # Fallback : on découvre les types présents dans le dictionnaire
            all_types = set()
            for k, t1, t2, dist in dict_pots.keys():
                all_types.add(t1)
                all_types.add(t2)
            self.sorted_types = [str(i) for i in sorted(list(all_types))]
        
        self.potential_tensor = torch.zeros(taille_mat, dtype=torch.float32).to(self.device)
        for (k, t1, t2, dist), energy in dict_pots.items():
            if k < taille_mat[0] and t1 < taille_mat[1] and t2 < taille_mat[2] and dist < taille_mat[3]:
                # On utilise directement les indices t1, t2 car ils sont déjà mappés 0-22 pour RASP
                self.potential_tensor[k, t1, t2, dist] = energy
                self.potential_tensor[k, t2, t1, dist] = energy

    def convert_pdb_to_rigid_tensors(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df_atoms = ppdb.df['ATOM'].copy()
        
        df_atoms['rasp_type'] = df_atoms.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name'], self.type_RASP), axis=1)
        self.df_filtered = df_atoms[df_atoms['rasp_type'] != -1].reset_index(drop=True)
        
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

        # PARAMÈTRES OPTIMISABLES : 
        # 1. Position du centre (translation)
        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        
        # 2. Angles de rotation (Euler : rx, ry, rz) initiaux à 0
        self.rot_angles = torch.nn.Parameter(torch.zeros((num_nucs, 3), dtype=torch.float32))
        
        # CONSTANTE : offsets internes des atomes par rapport à leur référence
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        # Identification Backbone
        self.bb_i_idx, self.bb_j_idx = [], [] # O3'(i) et P(i+1)
        self.p_curr_idx, self.p_next_idx = [], [] # P(i) et P(i+1)
        self.c3_idx = [] # C3'(i)

        for i in range(len(unique_res) - 1):
            res_curr, res_next = unique_res[i], unique_res[i+1]
            
            # On isole les atomes du résidu courant et suivant
            # (Astuce : filtrer une seule fois par résidu est plus rapide)
            df_curr = self.df_filtered[self.df_filtered['residue_number'] == res_curr]
            df_next = self.df_filtered[self.df_filtered['residue_number'] == res_next]
            
            idx_o3 = df_curr[df_curr['atom_name'] == "O3'"].index
            idx_c3 = df_curr[df_curr['atom_name'] == "C3'"].index
            idx_p_c = df_curr[df_curr['atom_name'] == "P"].index
            idx_p_n = df_next[df_next['atom_name'] == "P"].index
            
            # 1. Liaison O3'(i) - P(i+1)
            if not idx_o3.empty and not idx_p_n.empty:
                self.bb_i_idx.append(idx_o3[0])
                self.bb_j_idx.append(idx_p_n[0])
                
            # 2. Virtual Bond P(i) - P(i+1)
            if not idx_p_c.empty and not idx_p_n.empty:
                self.p_curr_idx.append(idx_p_c[0])
                self.p_next_idx.append(idx_p_n[0])
                
            # 3. Pour l'angle C3'(i)-O3'(i)-P(i+1), on vérifie que les 3 existent
            if not idx_c3.empty and not idx_o3.empty and not idx_p_n.empty:
                self.c3_idx.append(idx_c3[0])
        
        self.bb_i_idx = torch.tensor(self.bb_i_idx, dtype=torch.long, device=self.device)
        self.bb_j_idx = torch.tensor(self.bb_j_idx, dtype=torch.long, device=self.device)
        self.p_curr_idx = torch.tensor(self.p_curr_idx, dtype=torch.long, device=self.device)
        self.p_next_idx = torch.tensor(self.p_next_idx, dtype=torch.long, device=self.device)
        self.c3_idx = torch.tensor(self.c3_idx, dtype=torch.long, device=self.device)
        
        # Données pour RASP
        atom_types = torch.tensor(self.df_filtered['rasp_type'].values, dtype=torch.long).to(self.device)
        n_atoms = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=self.device)
        
        res_tensor = torch.tensor(res_ids, dtype=torch.long).to(self.device)
        sep = torch.abs(res_tensor[i_idx] - res_tensor[j_idx])
        mask_k = sep > 0 
        
        self.pair_i = i_idx[mask_k]
        self.pair_j = j_idx[mask_k]
        self.k_vals = torch.clamp(sep[mask_k] - 1, 0, 5)
        self.t1_vals = atom_types[self.pair_i]
        self.t2_vals = atom_types[self.pair_j]

    def get_rotation_matrices(self):
        """Convertit les 3 angles d'Euler en une matrice de rotation 3x3 pour chaque nucléotide."""
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])

        N = self.rot_angles.shape[0]

        Rx = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(self.device)
        Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = cos_a, -sin_a, sin_a, cos_a

        Ry = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(self.device)
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = cos_b, sin_b, -sin_b, cos_b

        Rz = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(self.device)
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = cos_g, -sin_g, sin_g, cos_g

        # Matrice finale R = Rz * Ry * Rx
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_current_full_coords(self):
        # 1. On récupère les matrices de rotation pour chaque nucléotide
        R = self.get_rotation_matrices() 
        
        # 2. On attribue à chaque atome la matrice de son nucléotide
        R_atoms = R[self.atom_to_nuc_idx] # (N_atoms, 3, 3)
        
        # 3. On applique la rotation aux offsets
        rotated_offsets = torch.bmm(R_atoms, self.offsets.unsqueeze(2)).squeeze(2)
        
        # 4. On ajoute les coordonnées de référence (translation)
        return self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets

    def setup_clash_detection(self):
        # On calcule la somme des rayons pour chaque paire possible
        # r_sum[i, j] = radius_i + radius_j
        r = self.vdw_radii_map[self.t1_vals.long()]
        r_j = self.vdw_radii_map[self.t2_vals.long()]
        self.min_dist_threshold = (r + r_j) * 0.85

    def _get_vdw_radii(self):
        # Valeurs standards simplifiées pour l'ARN (en Angströms)
        # On se base sur le premier caractère du type DFIRE (C, N, O, P)
        radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8}
        vdw_list = []
        for t in self.sorted_types:
            element = t[0] # Récupère 'C', 'N', etc.
            vdw_list.append(radii.get(element, 1.7))
        return torch.tensor(vdw_list, dtype=torch.float32).to(self.device)

    def calculate_detailed_scores(self, current_full_coords):
        p1 = current_full_coords[self.pair_i]
        p2 = current_full_coords[self.pair_j]
        
        # +1e-8 pour éviter le gradient NaN quand la distance est 0
        dists = torch.norm(p1 - p2, dim=1) + 1e-8
        max_dist_idx = self.potential_tensor.size(3) - 1
        # Soft mask pour interpolation (pas de changement de taille de tenseur !)
        d_clamp = torch.clamp(dists, 0.0, float(max_dist_idx)) 
        d0 = torch.floor(d_clamp).long()
        d1 = torch.clamp(d0 + 1, max=max_dist_idx)
        alpha = d_clamp - d0.float()
        
        energy0 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d0]
        energy1 = self.potential_tensor[self.k_vals, self.t1_vals, self.t2_vals, d1]
        
        interp_energy = (1 - alpha) * energy0 + alpha * energy1
        plateau_val = 2.7
        corrected_energy = interp_energy - plateau_val
        # On annule l'énergie des atomes trop éloignés
        valid_mask = (dists < float(max_dist_idx)).float()
        rasp_score = torch.sum(corrected_energy * valid_mask)

        # --- 1. Clash Stérique (Utilisation des rayons VDW préparés) ---
        p_i, p_j = current_full_coords[self.pair_i.long()], current_full_coords[self.pair_j.long()]
        dist_pairs = torch.norm(p_i - p_j, dim=1)

        # Utilise le seuil dynamique calculé dans setup_clash_detection
        thresholds = self.min_dist_threshold # Déjà sur le bon device
        clash_penalty = torch.sum(torch.clamp(thresholds - dist_pairs, min=0.0)**2) * self.clash_weight

        # Contrainte Backbone
        bb_penalty = torch.tensor(0.0, device=self.device)
        if len(self.bb_i_idx) > 0:
            # A. Liaison O3'-P (1.61 A)
            p_o3, p_p = current_full_coords[self.bb_i_idx], current_full_coords[self.bb_j_idx]
            dist_o3p = torch.norm(p_o3 - p_p, dim=1)
            bb_penalty += torch.sum((dist_o3p - 1.61)**2) * self.backbone_weight

            # B. Virtual Bond P-P (5.9 A) - Selon modèle SOP-RNA
            if len(self.p_curr_idx) > 0:
                dist_pp = torch.norm(current_full_coords[self.p_curr_idx] - current_full_coords[self.p_next_idx], dim=1)
                bb_penalty += torch.sum((dist_pp - 5.9)**2) * (self.backbone_weight * 0.5)

            # C. Angle C3'-O3'-P (cible ~120 deg soit cos = -0.5)
            p_c3 = current_full_coords[self.c3_idx]
            v1, v2 = p_c3 - p_o3, p_p - p_o3
            cos_angle = torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1)*torch.norm(v2, dim=1) + 1e-8)
            bb_penalty += torch.sum((cos_angle - (-0.5))**2) * 20.0
            
        return rasp_score, bb_penalty, clash_penalty

    def run_optimization(self):
        """
        Optimisation dynamique sans limite fixe d'époques ni de cycles.
        """
        # Initialisation de l'optimiseur
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
        
        best_ref_coords = self.ref_coords.clone().detach()
        best_rot_angles = self.rot_angles.clone().detach()
        self.best_score = float('inf')

        if self.verbose:
            print(f"🚀 Début de l'optimisation RASP dynamique...")
            print(f"Nombre de paires d'atomes à traiter : {self.pair_i.size(0):,}")
        
        current_noise_c = self.noise_coords
        current_noise_a = self.noise_angles
        cycles_sans_amelioration = 0
        cycle_count = 0

        # BOUCLE EXTERNE : Secousses (Exploration)
        while current_noise_c > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
            cycle_count += 1
            if self.verbose:
                print(f"\n--- Phase d'exploration {cycle_count} (Bruit: {current_noise_c:.4f}Å / {current_noise_a:.4f}rad) ---")
            
            patience_counter = 0
            prev_loss = float('inf')
            epoch = 0
            
            # BOUCLE INTERNE : Minimisation locale
            while True:
                optimizer.zero_grad()
                
                coords = self.get_current_full_coords()
                self.rasp_score, self.bb_penalty, self.clash_penalty = self.calculate_detailed_scores(coords)
                loss = self.rasp_score + self.bb_penalty + self.clash_penalty
                
                loss.backward()
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

                # Affichage tous les 100 pas
                if (epoch % 100 == 0) and self.verbose:
                    print(f"  Iteration {epoch:4d} | Total: {current_loss:.4f} | RASP: {self.rasp_score.item():.2f} | Backbone: {self.bb_penalty.item():.2f} | Clash: {self.clash_penalty.item():.2f}")

                if patience_counter >= self.patience_locale:
                    if self.verbose:
                        print(f"Minimum local atteint en {epoch} itérations.")
                    break
                
                if epoch > 10000:
                    if self.verbose:
                        print(f"Phase locale trop longue, arrêt de sécurité à 10000.")
                    break

                del coords, loss

            # --- BILAN DU CYCLE ---
            if current_loss < (self.best_score - self.min_delta):
                if self.verbose:
                    print(f"Nouveau record absolu ! {self.best_score:.4f} -> {current_loss:.4f}")
                self.best_score = current_loss
                best_ref_coords.copy_(self.ref_coords)
                best_rot_angles.copy_(self.rot_angles)
                # Sauvegarde des scores actuels pour les résumés finaux
                # Note: rasp_score, bb_penalty, clash_penalty sont déjà mis à jour dans la boucle
                cycles_sans_amelioration = 0
            else:
                cycles_sans_amelioration += 1
                if self.verbose:
                    print(f"Pas de record. (Échecs : {cycles_sans_amelioration}/{self.patience_globale})")

            # --- PRÉPARATION DU CYCLE SUIVANT ---
            current_noise_c *= self.taux_refroidissement
            current_noise_a *= self.taux_refroidissement
            
            if current_noise_c > self.bruit_min and cycles_sans_amelioration < self.patience_globale:
                if self.verbose:
                    print(f"Application du SHAKE. Retour à la meilleure conformation et ajout de bruit.")
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref_coords)
                    self.rot_angles.copy_(best_rot_angles)
                    self.ref_coords.add_(torch.randn_like(self.ref_coords) * current_noise_c)
                    self.rot_angles.add_(torch.randn_like(self.rot_angles) * current_noise_a)
                
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=self.lr)
                torch.cuda.empty_cache()

        if self.verbose:
            print("\n Optimisation terminée, restauration de la meilleure conformation trouvée...")
        with torch.no_grad():
            self.ref_coords.copy_(best_ref_coords)
            self.rot_angles.copy_(best_rot_angles)
            # Re-calcul des scores pour le reporting final
            coords = self.get_current_full_coords()
            self.rasp_score, self.bb_penalty, self.clash_penalty = self.calculate_detailed_scores(coords)
            
        self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_full_coords = self.get_current_full_coords().detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_full_coords
        out_ppdb.df["ATOM"]["chain_id"] = "A"
        out_ppdb.to_pdb(path=self.output_path)
        if self.verbose:
            print(f"Optimisation Rigide terminée. Meilleur score: {self.best_score:.4f}, RASP : {self.rasp_score:.4f}, Backbone : {self.bb_penalty:.4f}")
            print(f"Fichier sauvegardé : {self.output_path}")