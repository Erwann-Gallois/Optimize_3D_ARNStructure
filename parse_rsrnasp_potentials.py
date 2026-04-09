import os
import numpy as np
from biopandas.pdb import PandasPdb

# Exact ordered list of 85 atom types from "85atom_type.dat"
ATOM_TYPES_85 = [
    "AP", "AOP1", "AOP2", "AO5'", "AC5'", "AC4'", "AO4'", "AC3'", "AO3'", "AC2'", "AO2'", "AC1'", "AN9", "AC8", "AN7", "AC5", "AC6", "AN6", "AN1", "AC2", "AN3", "AC4",
    "UP", "UOP1", "UOP2", "UO5'", "UC5'", "UC4'", "UO4'", "UC3'", "UO3'", "UC2'", "UO2'", "UC1'", "UN1", "UC2", "UO2", "UN3", "UC4", "UO4", "UC5", "UC6",
    "CP", "COP1", "COP2", "CO5'", "CC5'", "CC4'", "CO4'", "CC3'", "CO3'", "CC2'", "CO2'", "CC1'", "CN1", "CC2", "CO2", "CN3", "CC4", "CN4", "CC5", "CC6",
    "GP", "GOP1", "GOP2", "GO5'", "GC5'", "GC4'", "GO4'", "GC3'", "GO3'", "GC2'", "GO2'", "GC1'", "GN9", "GC8", "GN7", "GC5", "GC6", "GO6", "GN1", "GC2", "GN2", "GN3", "GC4"
]

# Quick lookup dictionary (Type_String -> Index 0-84)
TYPE_TO_IDX = {t: i for i, t in enumerate(ATOM_TYPES_85)}

def get_rsrnasp_type(res_name, atom_name):
    """
    Combines residue name and atom name to find rsRNASP index (0 to 84).
    Returns -1 if the atom is not supported or is a hydrogen.
    """
    # Residue cleaning
    res_name = res_name.strip().upper()
    if res_name.startswith('R'): 
        res_name = res_name[1:] 
    if len(res_name) > 1 and res_name in ['ADE', 'URA', 'CYT', 'GUA']:
        res_name = res_name[0]
        
    # Atom cleaning (converts PDB O5* to O5' for example)
    atom_name = atom_name.strip().replace('*', "'")
    
    # Ignore hydrogen atoms
    if atom_name.startswith('H'): 
        return -1
        
    # Combination (e.g., "A" + "O5'" -> "AO5'")
    combo_name = res_name + atom_name
    
    return TYPE_TO_IDX.get(combo_name, -1)

def load_rsrnasp_potentials(short_path, long_path):
    """
    Loads both rsRNASP potential files.
    Returns: (num_types, num_dist_bins, dict_potentials)
    
    dict_potentials structure:
    Key = (k_state, t1, t2, dist_bin)
    Value = Energy (float)
    
    - k_state = 0 pour short-ranged (|i-j| <= 4)
    - k_state = 1 pour long-ranged (|i-j| >= 5)
    """
    if not os.path.exists(short_path) or not os.path.exists(long_path):
        raise FileNotFoundError(f"Files not found: check paths {short_path} and {long_path}")

    potentials = {}
    max_dist_bin = 0
    
    # Internal function to parse a specific file
    def parse_file(filepath, k_state):
        nonlocal max_dist_bin
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Check if the line has 4 columns (t1, t2, bin, energy)
                if len(parts) >= 4:
                    t1 = int(parts[0])
                    t2 = int(parts[1])
                    dist_bin = int(parts[2])
                    energy = float(parts[3])
                    
                    # Symmetry: interaction t1-t2 is equal to t2-t1
                    potentials[(k_state, t1, t2, dist_bin)] = energy
                    potentials[(k_state, t2, t1, dist_bin)] = energy
                    
                    if dist_bin > max_dist_bin:
                        max_dist_bin = dist_bin

    # 1. Loading Short-Range potential (index K = 0)
    parse_file(short_path, k_state=0)
    
    # 2. Loading Long-Range potential (index K = 1)
    parse_file(long_path, k_state=1)
    
    num_types = len(ATOM_TYPES_85) # Normally 85
    num_dist_bins = max_dist_bin + 1
    
    return num_types, num_dist_bins, potentials

def calculate_rsrnasp_score(pdb_path, potentials_dict, step_distance=0.3, cutoff_short=13.0, cutoff_long=24.0):
    """
    Utility function to evaluate a complete PDB statically.
    (Useful to check native structure energy without running the optimizer).
    """
    ppdb = PandasPdb().read_pdb(pdb_path)
    df = ppdb.df['ATOM']
    
    # Remove hydrogens
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
        atom1 = get_rsrnasp_type(res_names[i], atom_names[i])
        if atom1 == -1: 
            missing_atoms.add((res_names[i], atom_names[i]))
            continue
            
        for j in range(i + 1, n_atoms):
            atom2 = get_rsrnasp_type(res_names[j], atom_names[j])
            if atom2 == -1: 
                continue
                
            seq_sep = abs(res_ids[i] - res_ids[j])
            
            # rsRNASP ignores atoms of the same residue
            if seq_sep == 0: 
                continue 
            
            # Determining state K (Short vs Long range)
            if seq_sep <= 4:
                k_state = 0
                cutoff = cutoff_short
            else:
                k_state = 1
                cutoff = cutoff_long
            
            # Distance calculation
            dist_sq = np.sum((coords[i] - coords[j])**2)
            dist_angstrom = np.sqrt(dist_sq)
            
            # Cutoff application (no score if beyond)
            if dist_angstrom > cutoff:
                continue 
            
            # Physical distance to "Bin" conversion for dictionary lookup
            dist_bin = int(np.floor(dist_angstrom / step_distance))
            
            # Energy retrieval (default 0.0 if matrix slot doesn't exist)
            energy = potentials_dict.get((k_state, atom1, atom2, dist_bin), 0.0)
            
            total_energy += energy
            pairs_scored += 1
            
    if missing_atoms:
        print("Warning: some heavy atoms were not mapped for rsRNASP (they will be ignored):")
        print(missing_atoms)
            
    return total_energy, pairs_scored