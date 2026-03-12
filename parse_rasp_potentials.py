import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb

# Mapping exact d'après la table RASP (Types 1 à 23 -> Index 0 à 22 pour le code)
# Type 1: OP1, OP2, OP3
# Type 2: P
# Type 3: O5'
# Type 4: C5'
# Type 5: C4', C3', C2'
# Type 6: O2', O3'
# Type 7: C1'
# Type 8: O4'
# Type 9: N1(Pyr), N9(Pur)
# Type 10: C8(Pur)
# Type 11: N3, N7(Pur), N1(A), N3(C) -> Attention Pur = A, G; Pyr = C, U
# Type 12: C5(Pur)
# Type 13: C4(Pur)
# Type 14: C2(A)
# Type 15: C6(A), C4(C)
# Type 16: N6(A), N4(C), N2(G)
# Type 17: C2(G)
# Type 18: C6(G), C4(U)
# Type 19: O2(Pyr), O6(G), O4(U)
# Type 20: C2(Pyr)
# Type 21: C5(Pyr)
# Type 22: C6(Pyr)
# Type 23: N1(G), N3(U)

# On décale de -1 car les indices de la matrice RASP vont de 0 à 22 !
RASP_ATOM_TYPES = {
    # == BACKBONE (Tous résidus) ==
    **{(res, at): 0 for res in ['A','C','G','U'] for at in ['OP1', 'OP2', 'OP3']},
    **{(res, 'P'): 1 for res in ['A','C','G','U']},
    **{(res, "O5'"): 2 for res in ['A','C','G','U']},
    **{(res, "C5'"): 3 for res in ['A','C','G','U']},
    **{(res, at): 4 for res in ['A','C','G','U'] for at in ["C4'", "C3'", "C2'"]},
    **{(res, at): 5 for res in ['A','C','G','U'] for at in ["O2'", "O3'"]},
    **{(res, "C1'"): 6 for res in ['A','C','G','U']},
    **{(res, "O4'"): 7 for res in ['A','C','G','U']},
    
    # == BASES ==
    # Type 9 (idx 8) : N1(Pyr), N9(Pur) -> N1(C,U), N9(A,G)
    ('C', 'N1'): 8, ('U', 'N1'): 8, ('A', 'N9'): 8, ('G', 'N9'): 8,
    
    # Type 10 (idx 9) : C8(Pur) -> C8(A,G)
    ('A', 'C8'): 9, ('G', 'C8'): 9,
    
    # Type 11 (idx 10) : N3, N7(Pur), N1(A), N3(C) -> N3(A,G), N7(A,G), N1(A), N3(C)
    ('A', 'N3'): 10, ('G', 'N3'): 10, ('A', 'N7'): 10, ('G', 'N7'): 10, ('A', 'N1'): 10, ('C', 'N3'): 10,
    
    # Type 12 (idx 11) : C5(Pur) -> C5(A,G)
    ('A', 'C5'): 11, ('G', 'C5'): 11,
    
    # Type 13 (idx 12) : C4(Pur) -> C4(A,G)
    ('A', 'C4'): 12, ('G', 'C4'): 12,
    
    # Type 14 (idx 13) : C2(A)
    ('A', 'C2'): 13,
    
    # Type 15 (idx 14) : C6(A), C4(C)
    ('A', 'C6'): 14, ('C', 'C4'): 14,
    
    # Type 16 (idx 15) : N6(A), N4(C), N2(G)
    ('A', 'N6'): 15, ('C', 'N4'): 15, ('G', 'N2'): 15,
    
    # Type 17 (idx 16) : C2(G)
    ('G', 'C2'): 16,
    
    # Type 18 (idx 17) : C6(G), C4(U)
    ('G', 'C6'): 17, ('U', 'C4'): 17,
    
    # Type 19 (idx 18) : O2(Pyr), O6(G), O4(U) -> O2(C,U), O6(G), O4(U)
    ('C', 'O2'): 18, ('U', 'O2'): 18, ('G', 'O6'): 18, ('U', 'O4'): 18,
    
    # Type 20 (idx 19) : C2(Pyr) -> C2(C,U)
    ('C', 'C2'): 19, ('U', 'C2'): 19,
    
    # Type 21 (idx 20) : C5(Pyr) -> C5(C,U)
    ('C', 'C5'): 20, ('U', 'C5'): 20,
    
    # Type 22 (idx 21) : C6(Pyr) -> C6(C,U)
    ('C', 'C6'): 21, ('U', 'C6'): 21,
    
    # Type 23 (idx 22) : N1(G), N3(U)
    ('G', 'N1'): 22, ('U', 'N3'): 22
}


def load_rasp_potentials(filepath):
    """
    Charge le fichier de potentiel RASP (.nrg) dans un dictionnaire.
    Structure désirée : dict[(K, Type1, Type2, Dist)] = Energie
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Récupération de la taille de la matrice de potentiel
    taille = lines[0:2][1]
    taille_mat = taille.split("\t")
    last = taille_mat.pop(-1)
    last = last.split("\n")[0]
    taille_mat.append(last)
    taille_mat = list(map(int, taille_mat))
    taille_mat = tuple(taille_mat)
    header_idx = 2
    for i, line in enumerate(lines[:10]):
        if line.startswith("# K"):
            header_idx = i
            break

    rasp_dict = {}
    for line in lines[header_idx+1:]:
        if not line.strip(): continue
        parts = line.strip().split()
        if len(parts) == 5:
            k = int(parts[0])
            t1 = int(parts[1])
            t2 = int(parts[2])
            dist = int(parts[3])
            energy = float(parts[4])
            
            rasp_dict[(k, t1, t2, dist)] = energy
            rasp_dict[(k, t2, t1, dist)] = energy
    return taille_mat, rasp_dict

def get_rasp_type(res_name, atom_name, type_RASP="all"):
    """
    Retourne le type d'atome RASP (0-22). 
    Retourne -1 si atome non supporté.
    """
    res_name = res_name.strip().upper()
    if res_name.startswith('R'): res_name = res_name[1:] 
    if len(res_name) > 1 and res_name in ['ADE', 'URA', 'CYT', 'GUA']:
        res_name = res_name[0]
        
    atom_name = atom_name.strip()
    
    # Nettoyage atome PDB (souvent O5* au lieu de O5')
    atom_name = atom_name.replace('*', "'")

    if atom_name.startswith('H'): 
        return -1
        
    return RASP_ATOM_TYPES.get((res_name, atom_name), -1)

def calculer_score_rasp(pdb_path, potentials_dict):
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
        atom1 = get_rasp_type(res_names[i], atom_names[i])
        if atom1 == -1: 
            missing_atoms.add((res_names[i], atom_names[i]))
            continue
            
        for j in range(i + 1, n_atoms):
            atom2 = get_rasp_type(res_names[j], atom_names[j])
            if atom2 == -1: continue
                
            seq_sep = abs(res_ids[i] - res_ids[j])
            if seq_sep == 0: continue 
            
            k_idx = min(seq_sep - 1, 5) 
            
            dist_sq = np.sum((coords[i] - coords[j])**2)
            dist_angstrom = np.sqrt(dist_sq)
            
            # Les bins RASP sont par tranche de 1A (arrondi au sol)
            bin_dist = int(np.floor(dist_angstrom))
            
            if bin_dist > 19:
                continue 
                
            energy = potentials_dict.get((k_idx, atom1, atom2, bin_dist), 0.0)
            total_energy += energy
            pairs_scored += 1
            
    if missing_atoms:
        print("Avertissement : certains atomes lourds n'ont pas été mappés (ils seront ignorés) :")
        print(missing_atoms)
            
    return total_energy, pairs_scored