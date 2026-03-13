import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
import os

def load_dfire_potentials(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.loadtxt(filepath, usecols=range(2, 30))
        atom_pairs = np.loadtxt(filepath, usecols=(0, 1), dtype=str)
    potentials = {tuple(pair): row for pair, row in zip(atom_pairs, data)}
    return potentials
    
def get_dfire_type(atom_name, res_name):
    res_name = res_name.strip().upper()
    if res_name.startswith('R'): res_name = res_name[1:] 
    if len(res_name) > 1 and res_name in ['ADE', 'URA', 'CYT', 'GUA']:
        res_name = res_name[0]
    atom_name = atom_name.strip()
    atom_name = atom_name.replace('*', "'")
    if atom_name.startswith('H'): 
        return -1
    dfire_type = res_name + "_" + atom_name
    return dfire_type

def calculate_dfire_score(pdb_path, potentials):
    """
    Calcule le score énergétique global.
    """
    ppdb = PandasPdb().read_pdb(pdb_path)
    df = ppdb.df['ATOM']
    
    df = df[~df['atom_name'].str.startswith('H')].reset_index(drop=True)
    
    total_energy = 0.0
    pairs_scored = 0
    missing_atoms = set()
    
    coords = df[['x_coord', 'y_coord', 'z_coord']].values
    res_names = df['residue_name'].values
    atom_names = df['atom_name'].values
    res_ids = df['residue_number'].values
    
    n_atoms = len(df)
    
    for i in range(n_atoms):
        atom1 = get_dfire_type(atom_names[i], res_names[i])
        if atom1 == -1: 
            missing_atoms.add((atom_names[i], res_names[i]))
            continue
            
        for j in range(i + 1, n_atoms):
            atom2 = get_dfire_type(atom_names[j], res_names[j])
            energy_row = potentials[(atom1, atom2)]
            r = np.linalg.norm(coords[i] - coords[j])
            if r >= 19.6:
                continue
            idx = int(r / 0.7)   
            if idx < len(energy_row):
                energy = energy_row[idx]
            else:
                energy = 0.0 
            total_energy += energy
            pairs_scored += 1
            
    if missing_atoms:
        print("Avertissement : certains atomes lourds n'ont pas été mappés (ils seront ignorés) :")
        print(missing_atoms)

    return total_energy, pairs_scored