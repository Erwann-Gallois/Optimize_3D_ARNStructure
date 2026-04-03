import os
import numpy as np
from biopandas.pdb import PandasPdb

# La liste ordonnée exacte des 85 types d'atomes issue de "85atom_type.dat"
ATOM_TYPES_85 = [
    "AP", "AOP1", "AOP2", "AO5'", "AC5'", "AC4'", "AO4'", "AC3'", "AO3'", "AC2'", "AO2'", "AC1'", "AN9", "AC8", "AN7", "AC5", "AC6", "AN6", "AN1", "AC2", "AN3", "AC4",
    "UP", "UOP1", "UOP2", "UO5'", "UC5'", "UC4'", "UO4'", "UC3'", "UO3'", "UC2'", "UO2'", "UC1'", "UN1", "UC2", "UO2", "UN3", "UC4", "UO4", "UC5", "UC6",
    "CP", "COP1", "COP2", "CO5'", "CC5'", "CC4'", "CO4'", "CC3'", "CO3'", "CC2'", "CO2'", "CC1'", "CN1", "CC2", "CO2", "CN3", "CC4", "CN4", "CC5", "CC6",
    "GP", "GOP1", "GOP2", "GO5'", "GC5'", "GC4'", "GO4'", "GC3'", "GO3'", "GC2'", "GO2'", "GC1'", "GN9", "GC8", "GN7", "GC5", "GC6", "GO6", "GN1", "GC2", "GN2", "GN3", "GC4"
]

# Dictionnaire de recherche rapide (Type_String -> Index 0-84)
TYPE_TO_IDX = {t: i for i, t in enumerate(ATOM_TYPES_85)}

def get_rsrnasp_type(res_name, atom_name):
    """
    Combine le nom du résidu et de l'atome pour trouver l'index rsRNASP (0 à 84).
    Retourne -1 si l'atome n'est pas supporté ou est un hydrogène.
    """
    # Nettoyage du résidu
    res_name = res_name.strip().upper()
    if res_name.startswith('R'): 
        res_name = res_name[1:] 
    if len(res_name) > 1 and res_name in ['ADE', 'URA', 'CYT', 'GUA']:
        res_name = res_name[0]
        
    # Nettoyage de l'atome (convertit PDB O5* en O5' par ex)
    atom_name = atom_name.strip().replace('*', "'")
    
    # On ignore les atomes d'hydrogène
    if atom_name.startswith('H'): 
        return -1
        
    # Combinaison (ex: "A" + "O5'" -> "AO5'")
    combo_name = res_name + atom_name
    
    return TYPE_TO_IDX.get(combo_name, -1)

def load_rsrnasp_potentials(short_path, long_path):
    """
    Charge les deux fichiers de potentiels rsRNASP.
    Retourne : (num_types, num_dist_bins, dict_potentials)
    
    Structure de dict_potentials :
    Clé = (k_state, t1, t2, dist_bin)
    Valeur = Energy (float)
    
    - k_state = 0 pour short-ranged (|i-j| <= 4)
    - k_state = 1 pour long-ranged (|i-j| >= 5)
    """
    if not os.path.exists(short_path) or not os.path.exists(long_path):
        raise FileNotFoundError(f"Fichiers introuvables : vérifiez les chemins {short_path} et {long_path}")

    potentials = {}
    max_dist_bin = 0
    
    # Fonction interne pour parser un fichier spécifique
    def parse_file(filepath, k_state):
        nonlocal max_dist_bin
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # On vérifie que la ligne a bien 4 colonnes (t1, t2, bin, energy)
                if len(parts) >= 4:
                    t1 = int(parts[0])
                    t2 = int(parts[1])
                    dist_bin = int(parts[2])
                    energy = float(parts[3])
                    
                    # Symétrie : l'interaction t1-t2 est égale à t2-t1
                    potentials[(k_state, t1, t2, dist_bin)] = energy
                    potentials[(k_state, t2, t1, dist_bin)] = energy
                    
                    if dist_bin > max_dist_bin:
                        max_dist_bin = dist_bin

    # 1. Chargement du potentiel Short-Range (index K = 0)
    parse_file(short_path, k_state=0)
    
    # 2. Chargement du potentiel Long-Range (index K = 1)
    parse_file(long_path, k_state=1)
    
    num_types = len(ATOM_TYPES_85) # Normalement 85
    num_dist_bins = max_dist_bin + 1
    
    return num_types, num_dist_bins, potentials

def calculate_rsrnasp_score(pdb_path, potentials_dict, step_distance=0.3, cutoff_short=13.0, cutoff_long=24.0):
    """
    Fonction utilitaire pour évaluer un PDB complet de manière statique.
    (Pratique pour vérifier l'énergie d'une structure native sans lancer l'optimiseur).
    """
    ppdb = PandasPdb().read_pdb(pdb_path)
    df = ppdb.df['ATOM']
    
    # On retire les hydrogènes
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
            
            # rsRNASP ignore les atomes du même résidu
            if seq_sep == 0: 
                continue 
            
            # Détermination de l'état K (Short vs Long range)
            if seq_sep <= 4:
                k_state = 0
                cutoff = cutoff_short
            else:
                k_state = 1
                cutoff = cutoff_long
            
            # Calcul de la distance
            dist_sq = np.sum((coords[i] - coords[j])**2)
            dist_angstrom = np.sqrt(dist_sq)
            
            # Application du cutoff (on ne score pas si on est au-delà)
            if dist_angstrom > cutoff:
                continue 
            
            # Conversion de la distance physique en "Bin" pour lire le dictionnaire
            dist_bin = int(np.floor(dist_angstrom / step_distance))
            
            # Récupération de l'énergie (par défaut 0.0 si la case de la matrice n'existe pas)
            energy = potentials_dict.get((k_state, atom1, atom2, dist_bin), 0.0)
            
            total_energy += energy
            pairs_scored += 1
            
    if missing_atoms:
        print("Avertissement : certains atomes lourds n'ont pas été mappés rsRNASP (ils sont ignorés) :")
        print(missing_atoms)
            
    return total_energy, pairs_scored