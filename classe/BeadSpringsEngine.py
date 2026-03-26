import torch
import numpy as np
import classe.BaseEngine as BaseEngine
import classe.ModelContext as ModelContext
from biopandas.pdb import PandasPdb

class BeadSpringEngine(BaseEngine):
    def __init__(self, sequence: str, wca_sigma=5.0, fene_r0=6.0, **kwargs):
        super().__init__(**kwargs)
        self.sequence = sequence.upper()
        self.n_beads = len(sequence)
        self.wca_sigma = wca_sigma
        self.wca_cutoff = (2 ** (1 / 6)) * wca_sigma
        self.fene_r0 = fene_r0
        self.fene_k, self.fene_R0 = 30.0, 2.0
        
        init_coords = torch.zeros((self.n_beads, 3))
        init_coords[:, 0] = torch.arange(self.n_beads) * fene_r0
        init_coords += torch.randn_like(init_coords) * 0.5
        self.coords = torch.nn.Parameter(init_coords.to(self.device))
        self.res_ids = torch.arange(self.n_beads, device=self.device)

    def get_context(self) -> ModelContext:
        dist_matrix = torch.cdist(self.coords, self.coords) + 1e-8
        i, j = torch.triu_indices(self.n_beads, self.n_beads, offset=1, device=self.device)
        return ModelContext(
            coords=self.coords,
            res_ids=self.res_ids,
            res_names=list(self.sequence),
            dist_matrix=dist_matrix,
            pairwise_dists=dist_matrix[i, j],
            dists_adj=torch.diag(dist_matrix, diagonal=1)
        )

    def calculate_physical_penalties(self, ctx: ModelContext):
        d = ctx.pairwise_dists
        mask = (d < self.wca_cutoff)
        e_wca = torch.sum(4 * 1.0 * ((self.wca_sigma / d[mask])**12 - (self.wca_sigma / d[mask])**6) + 1.0) if mask.any() else 0.0
        delta_r = ctx.dists_adj - self.fene_r0
        ratio_sq = torch.clamp((delta_r / self.fene_R0) ** 2, max=0.999)
        e_fene = -0.5 * self.fene_k * (self.fene_R0 ** 2) * torch.sum(torch.log(1 - ratio_sq))
        return e_wca + e_fene

    def get_optimized_df(self) -> pd.DataFrame:
        """Retourne un DataFrame avec les positions finales des billes."""
        coords = self.coords.detach().cpu().numpy()
        df = pd.DataFrame({
            'res_id': self.res_ids.cpu().numpy(),
            'res_name': list(self.sequence),
            'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2]
        })
        return df

    def run_optimization(self, num_cycles=1, max_iter=100, lr=0.5):
        optimizer = torch.optim.LBFGS([self.coords], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        for cycle in range(num_cycles):
            def closure():
                optimizer.zero_grad()
                ctx = self.get_context()
                loss = self.calculate_total_loss(ctx) + self.calculate_physical_penalties(ctx)
                loss.backward()
                return loss
            optimizer.step(closure)
            final_loss = closure().item()
            if self.verbose: print(f"Cycle {cycle+1} | Loss: {final_loss:.4f}")
            if final_loss < self.best_score: self.best_score = final_loss
        return self.get_optimized_df(), self.best_score