import os
import torch
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
# import depuis votre script
from parse_dfire_potentials import load_dfire_potentials, get_dfire_type

class RNA_DFIRE_Rigid:
    def __init__(self, pdb_path, lr=0.2, output_path="output_rigid.pdb", ref_atom="C3'", num_cycles=5, epochs_per_cycle=100, noise_coords=1.5, noise_angles=0.5):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Le fichier PDB {pdb_path} n'existe pas.")
        
        self.pdb_path = pdb_path
        self.lr = lr
        self.output_path = output_path
        self.ref_atom = ref_atom
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.noise_coords = noise_coords
        self.noise_angles = noise_angles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device : {self.device}")
        self.best_score = float('inf')
        
        # 1. Chargement des potentiels
        self.load_dict_potentials()
        
        # 2. Chargement et préparation de la structure rigide
        self.convert_pdb_to_rigid_tensors(self.pdb_path)

    def load_dict_potentials(self):
        path = "potentials/matrice_dfire.dat"
        if os.path.exists(path):
            dict_pots = load_dfire_potentials(path)
            self.convert_dict_to_tensor(dict_pots)
        else:
            print(f"Fichier de potentiel non trouvé : {path}")

    def convert_dict_to_tensor(self, dict_pots):
        