import torch
import classe.ModelContext as ModelContext
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type

class DFIREScoreModule(torch.nn.Module):
    def __init__(self, potential_path: str, nbre_nt_exclu: int = 2):
        super().__init__()
        self.nbre_nt_exclu = nbre_nt_exclu
        self._load_potential(potential_path)

    def _load_potential(self, path):
        dict_pots = load_dfire_potentials(path)
        all_types = sorted(list(set(t for pair in dict_pots.keys() for t in pair)))
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        bins = len(next(iter(dict_pots.values())))
        self.pot_tensor = torch.nn.Parameter(torch.zeros((len(all_types), len(all_types), bins)), requires_grad=False)
        for (t1, t2), vals in dict_pots.items():
            i1, i2 = self.type_to_idx[t1], self.type_to_idx[t2]
            v = torch.tensor(vals)
            self.pot_tensor[i1, i2] = v; self.pot_tensor[i2, i1] = v
        self.get_type_fn = get_dfire_type

    def forward(self, ctx: ModelContext) -> torch.Tensor:
        types = [self.get_type_fn(an, rn) for an, rn in zip(ctx.atom_names, ctx.res_names)]
        type_indices = torch.tensor([self.type_to_idx.get(t, -1) for t in types], device=ctx.coords.device)
        n = len(ctx.coords)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=ctx.coords.device)
        sep = torch.abs(ctx.res_ids[i_idx] - ctx.res_ids[j_idx])
        mask = (sep > self.nbre_nt_exclu) & (type_indices[i_idx] != -1) & (type_indices[j_idx] != -1)
        dists = torch.norm(ctx.coords[i_idx[mask]] - ctx.coords[j_idx[mask]], dim=1) + 1e-8
        d_scaled = dists / 0.7
        d0 = torch.floor(torch.clamp(d_scaled, 0, 27)).long()
        d1 = torch.clamp(d0 + 1, max=27)
        alpha = d_scaled - d0.float()
        t1, t2 = type_indices[i_idx[mask]], type_indices[j_idx[mask]]
        e0 = self.pot_tensor[t1, t2, d0]
        e1 = self.pot_tensor[t1, t2, d1]
        energy = (1 - alpha) * e0 + alpha * e1
        return torch.sum(energy * (dists < 19.6).float())
