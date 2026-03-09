import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
# Assure-toi que ces imports fonctionnent avec tes fichiers locaux
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class RNA_RASP_Gradient:
    def __init__(self, pdb_path, lr=0.1, epoch=100, history=10, type_RASP="all", output_path="output.pdb"):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.epoch = epoch
        self.history = history
        self.type_RASP = type_RASP
        self.output_path = output_path
        self.potential_tensor = None
        self.best_score = float('inf') # Initialisation ici
        
        self.load_dict_potentials()
        self.convert_pdb_to_tensor(self.pdb_path)

    def load_dict_potentials(self):
        path = f"potentials/{self.type_RASP}.nrg"
        if os.path.exists(path):
            dict_pots = load_rasp_potentials(path)
            self.dict_potentials = dict_pots
            self.convert_dict_to_tensor()
        else:
            print(f"⚠️ Fichier de potentiel non trouvé : {path}")

    def convert_dict_to_tensor(self):
        self.potential_tensor = torch.zeros((6, 23, 23, 20), dtype=torch.float32)
        for (k, t1, t2, dist), energy in self.dict_potentials.items():
            if k < 6 and t1 < 23 and t2 < 23 and dist < 20:
                self.potential_tensor[k, t1, t2, dist] = energy
                self.potential_tensor[k, t2, t1, dist] = energy

    def convert_pdb_to_tensor(self, pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        df = ppdb.df['ATOM']
        
        df['rasp_type'] = df.apply(lambda row: get_rasp_type(row['residue_name'], row['atom_name']), axis=1)
        self.df_filtered = df[df['rasp_type'] != -1].reset_index(drop=True)
        
        # AJOUT DE .contiguous() ICI pour éviter l'erreur L-BFGS
        raw_coords = torch.tensor(self.df_filtered[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
        self.coords = torch.nn.Parameter(raw_coords.contiguous())
        
        atom_types = torch.tensor(self.df_filtered['rasp_type'].values, dtype=torch.long)
        res_ids = torch.tensor(self.df_filtered['residue_number'].values, dtype=torch.long)
        
        n = len(self.df_filtered)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        
        sep = torch.abs(res_ids[i_idx] - res_ids[j_idx])
        mask_k = sep > 0 
        
        self.i_idx = i_idx[mask_k]
        self.j_idx = j_idx[mask_k]
        self.k_vals = torch.clamp(sep[mask_k] - 1, 0, 5)
        self.t1_vals = atom_types[self.i_idx]
        self.t2_vals = atom_types[self.j_idx]

    def calculate_total_score(self, current_coords):
        p1 = current_coords[self.i_idx]
        p2 = current_coords[self.j_idx]
        dists = torch.sqrt(torch.sum((p1 - p2)**2, dim=1) + 1e-8)
        
        mask_dist = dists < 19.0
        if not mask_dist.any():
            return torch.tensor(0.0, requires_grad=True)
            
        d_sub = dists[mask_dist]
        k_sub = self.k_vals[mask_dist]
        t1_sub = self.t1_vals[mask_dist]
        t2_sub = self.t2_vals[mask_dist]

        d0 = torch.floor(d_sub).long()
        d1 = torch.clamp(d0 + 1, max=19)
        alpha = d_sub - d0.float()
        
        energy0 = self.potential_tensor[k_sub, t1_sub, t2_sub, d0]
        energy1 = self.potential_tensor[k_sub, t1_sub, t2_sub, d1]
        
        return torch.sum((1 - alpha) * energy0 + alpha * energy1)

    def perturb_coords(self, intensity=0.1):
        with torch.no_grad():
            self.coords.add_(torch.randn_like(self.coords) * intensity)
            # On s'assure que les données restent contiguës après modification
            self.coords.data = self.coords.data.contiguous()

    def bond_loss(self, current_coords):
        # p1 et p2 sont des vecteurs d'atomes consécutifs
        p1 = current_coords[:-1]
        p2 = current_coords[1:]
        dists = torch.norm(p1 - p2, dim=1)
        target = 1.55 # Distance moyenne en Angströms
        return torch.mean((dists - target)**2) * 1000.0 # Coefficient de raideur élevé

    def prepare_backbone_indices(self):
        """
        Identifie les paires d'atomes qui forment des liaisons covalentes 
        dans le backbone pour leur appliquer une contrainte stricte.
        """
        df = self.df_filtered
        bonds = []
        # Parcourt les atomes pour trouver les liaisons standards (ex: P-OP1, C4'-C3', etc.)
        # Une méthode simple : lier les atomes proches (< 1.7A) dans le PDB initial
        coords_init = self.coords.detach()
        dist_matrix = torch.cdist(coords_init, coords_init)
        
        # On crée un masque : distance < 1.7 Angström et atomes différents
        adj = (dist_matrix < 1.7) & (dist_matrix > 0.1)
        self.bond_pairs = torch.nonzero(adj) # Indices des atomes liés

    def backbone_loss(self, current_coords):
        if not hasattr(self, 'bond_pairs'):
            self.prepare_backbone_indices()
        
        p1 = current_coords[self.bond_pairs[:, 0]]
        p2 = current_coords[self.bond_pairs[:, 1]]
        dists = torch.norm(p1 - p2, dim=1)
        
        # On force ces distances à rester identiques à celles du PDB d'origine
        # (ou à une valeur moyenne de 1.5A)
        return torch.mean((dists - 1.5)**2) * 2000.0

    def run_optimization(self, num_cycles=3):
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        
        log_dir = f"logs/gradient_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logs: {log_dir}")
        
        # On définit l'optimiseur ICI pour être sûr qu'il prend les bonnes coords contiguës
        optimizer = torch.optim.LBFGS([self.coords], lr=self.lr, history_size=self.history, line_search_fn='strong_wolfe')
        self.global_step = 0

        def closure():
            optimizer.zero_grad()
            rasp_score = self.calculate_total_score(self.coords)
            bond_l = self.bond_loss(self.coords)
            bb_l = self.backbone_loss(self.coords)
            loss = rasp_score + bond_l + bb_l
            loss.backward()
            
            # Calcul de la norme du gradient pour diagnostic
            grad_norm = 0.0
            if self.coords.grad is not None:
                grad_norm = self.coords.grad.norm().item()
            
            # Logging TensorBoard
            writer.add_scalar("Loss/Total", loss.item(), self.global_step)
            writer.add_scalar("Loss/RASP", rasp_score.item(), self.global_step)
            writer.add_scalar("Loss/Bond", bond_l.item(), self.global_step)
            writer.add_scalar("Loss/Backbone", bb_l.item(), self.global_step)
            writer.add_scalar("Info/Gradient_Norm", grad_norm, self.global_step)
            self.global_step += 1
            
            return loss

        best_coords = self.coords.clone().detach()

        for cycle in range(num_cycles):
            print(f"Cycle {cycle+1}/{num_cycles}")
            writer.add_scalar("Info/Cycle", cycle + 1, self.global_step)
            for _ in range(self.epoch // num_cycles):
                loss = optimizer.step(closure)
                if loss.item() < self.best_score:
                    self.best_score = loss.item()
                    best_coords = self.coords.clone().detach()
                    print(f"  Nouveau meilleur score : {self.best_score:.4f}")
            
            if cycle < num_cycles - 1:
                self.perturb_coords(0.15)

        with torch.no_grad():
            self.coords.copy_(best_coords)
            
        writer.close()
        self.save_optimized_pdb()

    def save_optimized_pdb(self):
        final_coords = self.coords.detach().cpu().numpy()
        out_ppdb = PandasPdb()
        out_ppdb.df['ATOM'] = self.df_filtered.copy()
        out_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = final_coords
        out_ppdb.to_pdb(path=self.output_path)
        print(f"✅ Terminé. Meilleur Score RASP: {self.best_score:.4f}")
        print(f"📁 Fichier sauvegardé : {self.output_path}")