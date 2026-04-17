# core/topology.py
import torch
from biopandas.pdb import PandasPdb

# 1. Dictionnaires Universels (Alphabet de base)
RESIDUES = ["A", "C", "G", "U"]
RES_TO_ID = {res: i for i, res in enumerate(RESIDUES)}
NUM_RES = len(RESIDUES)

ATOMS = [
    # Backbone
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 
    # Bases
    "N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4", "N6", "O6", "N2", "N4", "O4"
]
ATOM_TO_ID = {atom: i for i, atom in enumerate(ATOMS)}
NUM_ATOMS = len(ATOMS)

class RNATopology:
    """Parse un PDB et le convertit en tenseurs d'identifiants universels."""
    
    def __init__(self, pdb_path, device="cpu"):
        self.device = device
        self.pdb_path = pdb_path
        self._parse_pdb()
        
    def _parse_pdb(self):
        ppdb = PandasPdb().read_pdb(self.pdb_path)
        df = ppdb.df['ATOM'].copy()
        
        # Nettoyage classique
        df = df[~df['atom_name'].str.startswith('H')].reset_index(drop=True)
        df['atom_name'] = df['atom_name'].str.strip().str.replace('*', "'", regex=False)
        
        def clean_res(res):
            res = res.strip().upper()
            if res.startswith('R'): res = res[1:]
            if len(res) > 1 and res in ['ADE', 'URA', 'CYT', 'GUA']:
                return res[0]
            return res
            
        df['residue_name'] = df['residue_name'].apply(clean_res)
        
        # Mapping vers nos entiers universels
        df['res_id'] = df['residue_name'].map(RES_TO_ID).fillna(-1).astype(int)
        df['atom_id'] = df['atom_name'].map(ATOM_TO_ID).fillna(-1).astype(int)
        
        # Rejet des atomes modifiés ou inconnus
        rejected = df[(df['res_id'] == -1) | (df['atom_id'] == -1)]
        if len(rejected) > 0:
            print(f"RNATopology Warning: {len(rejected)} atomes ignorés (non standards).")
            
        self.df_valid = df[(df['res_id'] != -1) & (df['atom_id'] != -1)].reset_index(drop=True)
        
        # Tenseurs finaux sur GPU/CPU
        self.coords = torch.tensor(self.df_valid[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32, device=self.device)
        self.seq_ids = torch.tensor(self.df_valid['residue_number'].values, dtype=torch.long, device=self.device)
        self.res_type_ids = torch.tensor(self.df_valid['res_id'].values, dtype=torch.long, device=self.device)
        self.atom_type_ids = torch.tensor(self.df_valid['atom_id'].values, dtype=torch.long, device=self.device)
        
        # On garde les données PDB brutes pour la sauvegarde à la fin
        self.df_pdb_data = self.df_valid.copy()