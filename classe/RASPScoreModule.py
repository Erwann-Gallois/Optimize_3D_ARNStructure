import torch
import classe.ModelContext as ModelContext
from parse_rasp_potentials import load_rasp_potentials, get_rasp_type

class RASPScoreModule(torch.nn.Module):
    """
    Module PyTorch encapsulant le champ de force RASP (Ribonucleic Acids Statistical Potential),
    un potentiel statistique plus complet/spécialisé pour l'ARN.
    """
    def __init__(self, type_RASP="all", nbre_nt_exclu: int = 2):
        """
        Args:
            type_RASP: Le sous-type de matrice énergétique (défaut 'all') utilisé.
            nbre_nt_exclu: Évite de scorer les atomes qui participent au voisinage chimique de même résidus / proches.
        """
        super().__init__()
        self.type_RASP = type_RASP
        self.nbre_nt_exclu = nbre_nt_exclu
        self._load_potential()

    def _load_potential(self):
        """
        Procède au chargement d'un fichier binaire .nrg propre à RASP 
        et le convertit en Tenseur PyTorch prêt à être interpolé et lu.
        """
        path = f"potentials/{self.type_RASP}.nrg"
        taille, dict_pots = load_rasp_potentials(path)
        self.pot_tensor = torch.nn.Parameter(torch.zeros(taille), requires_grad=False)
        for (k, t1, t2, d), energy in dict_pots.items():
            self.pot_tensor[k, t1, t2, d] = energy
            self.pot_tensor[k, t2, t1, d] = energy
        self.get_type_fn = get_rasp_type

    def forward(self, ctx: ModelContext) -> torch.Tensor:
        """
        Génère une indexation spatiale efficace pour RASP, et pour toutes paires applicables,
        lit et inter-pole bilinéairement son score. Produit l'énergie globale RASP (Tenseur à 1D).
        """
        types = torch.tensor([self.get_type_fn(rn, an, self.type_RASP) for rn, an in zip(ctx.res_names, ctx.atom_names)], device=ctx.coords.device)
        n = len(ctx.coords)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=ctx.coords.device)
        sep = torch.abs(ctx.res_ids[i_idx] - ctx.res_ids[j_idx])
        mask = (sep > self.nbre_nt_exclu) & (types[i_idx] != -1) & (types[j_idx] != -1)
        ii, jj = i_idx[mask], j_idx[mask]
        dists = torch.norm(ctx.coords[ii] - ctx.coords[jj], dim=1) + 1e-8
        k_vals = torch.clamp(sep[mask] - 1, 0, 5).long()
        max_d = self.pot_tensor.size(3) - 1
        d0 = torch.floor(torch.clamp(dists, 0, max_d)).long()
        d1 = torch.clamp(d0 + 1, max=max_d)
        alpha = dists - d0.float()
        e0 = self.pot_tensor[k_vals, types[ii], types[jj], d0]
        e1 = self.pot_tensor[k_vals, types[ii], types[jj], d1]
        return torch.sum(((1-alpha)*e0 + alpha*e1 - 2.7) * (dists < max_d).float())