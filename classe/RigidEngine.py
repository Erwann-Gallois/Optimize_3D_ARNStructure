import torch
import numpy as np
from classe.BaseEngine import BaseEngine
from classe.ModelContext import ModelContext
from biopandas.pdb import PandasPdb


class RigidEngine(BaseEngine):
    """
    Moteur d'optimisation basé sur des "corps rigides".
    Plutôt que d'optimiser les atomes indépendamment de manière libre, ce moteur
    préserve la géométrie locale (rigide) de chaque nucléotide, et optimise
    uniquement leurs translations et rotations globales dans l'espace.
    Cela préserve les géométries internes des bases tout en permettant l'assemblage 3D.
    """
    def __init__(self, pdb_path: str, ref_atom: str = "C3'", 
                 backbone_weight: float = 100.0, clash_weight: float = 50.0, noise_angles : float = 0.5, noise_coords : float = 1.5, **kwargs):
        """
        Initialise le moteur à corps rigides à partir d'un fichier PDB.
        
        Args:
            pdb_path: Chemin vers le fichier PDB initial.
            ref_atom: Atome de référence servant de centre de rotation (ex: C3').
            backbone_weight: Poids multiplicateur pour la fermeture du squelette.
            clash_weight: Poids des pénalités stériques (répulsion de van der Waals).
            **kwargs: Arguments passés au BaseEngine.
        """
        super().__init__(**kwargs)
        self.pdb_path = pdb_path
        self.ref_atom = ref_atom
        self.backbone_weight = backbone_weight
        self.clash_weight = clash_weight
        self.noise_angles = noise_angles
        self.noise_coords = noise_coords
        self._prepare_structure()
        self._setup_backbone_constraints()

    def _prepare_structure(self):
        """
        Extrait les coordonnées du fichier PDB originel et les décompose
        en centres de référence (par résidu) et en vecteurs de décalages (offsets) fixes
        pour chaque atome de ce même résidu.
        """
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        df = ppdb.df['ATOM'].copy()
        
        # Coordonnées initiales
        raw_coords = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, 
                                 dtype=torch.float32, device=self.device)
        self.res_ids = df['residue_number'].values
        unique_res = np.unique(self.res_ids)
        res_to_idx = {res: i for i, res in enumerate(unique_res)}
        self.atom_to_nuc_idx = torch.tensor([res_to_idx[r] for r in self.res_ids], device=self.device)
        
        # Initialisation des centres de rotation
        ref_coords_init = torch.zeros((len(unique_res), 3), device=self.device)
        for i, res in enumerate(unique_res):
            mask = (df['residue_number'] == res) & (df['atom_name'] == self.ref_atom)
            ref_coords_init[i] = raw_coords[mask.idxmax()] if mask.any() else raw_coords[(df['residue_number'] == res).idxmax()]

        self.ref_coords = torch.nn.Parameter(ref_coords_init.contiguous())
        self.rot_angles = torch.nn.Parameter(torch.zeros((len(unique_res), 3), device=self.device))
        self.offsets = raw_coords - ref_coords_init[self.atom_to_nuc_idx]
        
        self.df_atoms = df
        self.vdw_radii = torch.tensor([{'C':1.7,'N':1.55,'O':1.52,'P':1.8}.get(n[0],1.7) 
                                      for n in df['atom_name'].values], device=self.device)

    def _setup_backbone_constraints(self):
        """
        Identifie les indices (positions des atomes dans les tenseurs) pour pénaliser 
        ultérieurement les déformations du Backbone de l'ARN (i.e., la chaîne principale O3'-P(next), 
        la distance P(curr)-P(next), et l'angle C3'-O3'-P(next)).
        """
        self.bb_indices = {
            "o3": [], "p_next": [],  # Pour la distance de la liaison covalente O3' - P(next)
            "p_curr": [],            # Pour la distance P(curr) - P(next)
            "c3": []                 # Pour l'angle C3' - O3' - P(next)
        }
        
        unique_res = np.unique(self.res_ids)
        for i in range(len(unique_res) - 1):
            res_curr, res_next = unique_res[i], unique_res[i+1]
            df_curr = self.df_atoms[self.df_atoms['residue_number'] == res_curr]
            df_next = self.df_atoms[self.df_atoms['residue_number'] == res_next]
            
            idx_o3 = df_curr[df_curr['atom_name'] == "O3'"].index
            idx_c3 = df_curr[df_curr['atom_name'] == "C3'"].index
            idx_p_c = df_curr[df_curr['atom_name'] == "P"].index
            idx_p_n = df_next[df_next['atom_name'] == "P"].index
            
            # Liaison O3' - P(next)
            if not idx_o3.empty and not idx_p_n.empty:
                self.bb_indices["o3"].append(idx_o3[0])
                self.bb_indices["p_next"].append(idx_p_n[0])
                
                # Angle C3' - O3' - P(next) (seulement si O3' et P_n existent déjà)
                if not idx_c3.empty:
                    self.bb_indices["c3"].append(idx_c3[0])
            
            # Distance P(curr) - P(next)
            if not idx_p_c.empty and not idx_p_n.empty:
                self.bb_indices["p_curr"].append(idx_p_c[0])
                # Note: p_next pour cette paire est déjà géré par l'indexation de idx_p_n

        # Conversion en tenseurs
        for k in self.bb_indices:
            self.bb_indices[k] = torch.tensor(self.bb_indices[k], dtype=torch.long, device=self.device)

    def get_rotation_matrices(self):
        """
        Construit et retourne les matrices de rotation 3D complètes (Rz * Ry * Rx) 
        pour chaque corps rigide en fonction de leurs paramètres d'angles (d'Euler) optimisés
        (alpha, beta, gamma).
        """
        cos_a, sin_a = torch.cos(self.rot_angles[:, 0]), torch.sin(self.rot_angles[:, 0])
        cos_b, sin_b = torch.cos(self.rot_angles[:, 1]), torch.sin(self.rot_angles[:, 1])
        cos_g, sin_g = torch.cos(self.rot_angles[:, 2]), torch.sin(self.rot_angles[:, 2])
        N = self.rot_angles.shape[0]
        Rx = torch.eye(3, device=self.device).repeat(N, 1, 1)
        Ry = Rx.clone(); Rz = Rx.clone()
        Rx[:,1,1], Rx[:,1,2], Rx[:,2,1], Rx[:,2,2] = cos_a, -sin_a, sin_a, cos_a
        Ry[:,0,0], Ry[:,0,2], Ry[:,2,0], Ry[:,2,2] = cos_b, sin_b, -sin_b, cos_b
        Rz[:,0,0], Rz[:,0,1], Rz[:,1,0], Rz[:,1,1] = cos_g, -sin_g, sin_g, cos_g
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

    def get_context(self) -> ModelContext:
        """
        Assemble les coordonnées absolues actualisées de l'ensemble des atomes du modèle.
        Il applique la matrice de rotation courante suivie du vecteur de translation actuel 
        (le centre de référence optimisé).
        """
        R = self.get_rotation_matrices()[self.atom_to_nuc_idx]
        rotated_offsets = torch.bmm(R, self.offsets.unsqueeze(2)).squeeze(2)
        curr_coords = self.ref_coords[self.atom_to_nuc_idx] + rotated_offsets
        return ModelContext(
            coords=curr_coords, 
            res_ids=torch.tensor(self.res_ids, device=self.device),
            atom_names=list(self.df_atoms['atom_name']),
            res_names=list(self.df_atoms['residue_name'])
        )

    def calculate_physical_penalties(self, ctx: ModelContext):
        """
        Évalue toutes les contraintes de nature "physique" pour contraindre le modèle à avoir
        des longueurs et angles de liaison physiquement réalistes aux articulations.
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        # 1. Clashes Stériques : Pénalise fortement deux atomes qui se superposent (plus proches que vdw_radii)
        dist_mat = torch.cdist(ctx.coords, ctx.coords)
        i, j = torch.triu_indices(len(ctx.coords), len(ctx.coords), offset=1, device=self.device)
        
        # Exclure les atomes du même nucléotide car ils sont fixes (corps rigide)
        mask_diff_res = self.atom_to_nuc_idx[i] != self.atom_to_nuc_idx[j]
        i_filtered = i[mask_diff_res]
        j_filtered = j[mask_diff_res]
        
        dists = dist_mat[i_filtered, j_filtered]
        thresholds = (self.vdw_radii[i_filtered] + self.vdw_radii[j_filtered]) * 0.85
        clash_penalty = torch.sum(torch.clamp(thresholds - dists, min=0.0)**2) * self.clash_weight
        penalty += clash_penalty
        
        bb_penalty = torch.tensor(0.0, device=self.device)
        # 2. Backbone
        # O3' - P(next)
        if len(self.bb_indices["o3"]) > 0:
            p_o3 = ctx.coords[self.bb_indices["o3"]]
            p_pn = ctx.coords[self.bb_indices["p_next"]]
            d_o3p = torch.norm(p_o3 - p_pn, dim=1)
            bb_penalty += torch.sum((d_o3p - 1.61)**2) * self.backbone_weight
            
            # Angle C3' - O3' - P(next)
            if len(self.bb_indices["c3"]) > 0:
                p_c3 = ctx.coords[self.bb_indices["c3"]]
                v1, v2 = p_c3 - p_o3, p_pn - p_o3
                cos_ang = torch.sum(v1*v2, dim=1) / (torch.norm(v1, dim=1)*torch.norm(v2, dim=1) + 1e-8)
                bb_penalty += torch.sum((cos_ang - (-0.5))**2) * 20.0 # Poids fixe pour l'angle
        
        # P(curr) - P(next)
        if len(self.bb_indices["p_curr"]) > 0:
            # On récupère p_next correspondant (attention aux longueurs de listes si P est manquant)
            # Pour la simplicité, on recalcule la distance P-P si les deux indices existent
            p_pc = ctx.coords[self.bb_indices["p_curr"]]
            p_pn_dist = ctx.coords[self.bb_indices["p_next"][:len(p_pc)]] # Hypothèse de correspondance
            d_pp = torch.norm(p_pc - p_pn_dist, dim=1)
            bb_penalty += torch.sum((d_pp - 5.9)**2) * (self.backbone_weight * 0.5)
            
        penalty += bb_penalty
        return penalty, clash_penalty, bb_penalty

    def save_optimized_pdb(self, output_path: str = "output_optimized.pdb"):
        """Exporte l'état courant dans un fichier PDB via biopandas."""
        final_coords = self.get_context().coords.detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_atoms.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        out_ppdb.to_pdb(path=output_path, records=['ATOM'], gz=False, append_newline=True)
        if self.verbose: print(f"  💾 Structure sauvegardée dans : {output_path}")
        return out_ppdb, self.best_score

    def run_optimization(self, num_cycles=5, epochs=100, lr=0.2, noise_coords=1.5):
        """
        Boucle d'optimisation via l'algorithme Adam. Minimise l'énergie globale composée 
        du score statistique et des pénalités physiques.
        Intègre une approche rudimentaire de Simulated Annealing entre les cycles
        en perturbant fortement, puis de moins en moins, les positions de référence pour s'échapper
        des minimums locaux d'énergie.
        
        Args:
            num_cycles: Le nombre de fois que le processus de refroidissement démarre.
            epochs: Le nombre de pas de l'optimiseur par cycle.
            lr: Pas d'apprentissage de base pour l'algorithme Adam.
            noise_coords: Amplitude maximale du bruit blanc de perturbation à inter-cycle.
        """
        # On optimise à la fois le centre de masse (translations) et les rotations pour chaque corps (nucléotide)
        optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=lr)
        best_ref = self.ref_coords.clone().detach()
        best_rot = self.rot_angles.clone().detach()

        for cycle in range(num_cycles):
            for step in range(epochs):
                optimizer.zero_grad()
                ctx = self.get_context()
                bio_loss = self.calculate_total_loss(ctx)
                phys_loss, clash_loss, bb_loss = self.calculate_physical_penalties(ctx)
                loss = bio_loss + phys_loss
                loss.backward()
                optimizer.step()

                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_ref.copy_(self.ref_coords); best_rot.copy_(self.rot_angles)
                
                if step % 50 == 0 and self.verbose:
                    print(f"Cycle {cycle+1} | Step {step} | Loss: {loss.item():.2f} | Bio: {bio_loss.item():.2f} | Backbone: {bb_loss.item():.2f} | Clash: {clash_loss.item():.2f}")

            if cycle < num_cycles - 1:
                decay = 1.0 - (cycle / num_cycles)
                if self.verbose: print(f"  [!] SHAKE: Injection de bruit (decay={decay:.2f})")
                with torch.no_grad():
                    self.ref_coords.copy_(best_ref + torch.randn_like(best_ref) * self.noise_coords * decay)
                    self.rot_angles.copy_(best_rot + torch.randn_like(best_rot) * self.noise_angles * decay)
                optimizer = torch.optim.Adam([self.ref_coords, self.rot_angles], lr=lr)
        with torch.no_grad():
            self.ref_coords.copy_(best_ref); self.rot_angles.copy_(best_rot)
        return self.save_optimized_pdb()