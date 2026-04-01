import numpy as np
import os
import csv
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def extract_individual_distances(folder_path, output_path, ref_atom="C3'"):
    # Liste pour stocker chaque mesure individuelle
    # Format : [Nom_Fichier, Chaine, Index_Depart, Distance]
    rows_individual_distances = []
    
    total_files = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".cif"):
            continue
            
        nom_fichier = filename.split(".")[0]
        total_files += 1
        
        try:
            data_dict = MMCIF2Dict(os.path.join(folder_path, filename))

            atom_names = data_dict["_atom_site.label_atom_id"]
            alt_ids = data_dict["_atom_site.label_alt_id"]
            asym_ids = data_dict["_atom_site.label_asym_id"]
            model_nums = data_dict["_atom_site.pdbx_PDB_model_num"]
            x_coords = data_dict["_atom_site.Cartn_x"]
            y_coords = data_dict["_atom_site.Cartn_y"]
            z_coords = data_dict["_atom_site.Cartn_z"]

            chaines_uniques = sorted(list(set(asym_ids)))
            seq_ids = data_dict["_atom_site.label_seq_id"]
            for chaine in chaines_uniques:
                indices = [
                    i for i, name in enumerate(atom_names)
                    if name == ref_atom
                    and asym_ids[i] == chaine 
                    and model_nums[i] == "1" 
                    and alt_ids[i] in [".", "A"]
                ]
                
                if len(indices) >= 2:
                    # Conversion des coordonnées en float
                    coords = np.array([
                        [float(x_coords[i]), float(y_coords[i]), float(z_coords[i])]
                        for i in indices
                    ])
                    
                    # Calcul des distances point à point
                    for j in range(len(coords) - 1):
                        idx1 = indices[j]
                        idx2 = indices[j+1]
                        
                        # VÉRIFICATION DE LA CONTINUITÉ
                        s1 = int(seq_ids[idx1])
                        s2 = int(seq_ids[idx2])
                        
                        if s2 == s1 + 1:  # On ne calcule que si les résidus se suivent
                            p1 = np.array([float(x_coords[idx1]), float(y_coords[idx1]), float(z_coords[idx1])])
                            p2 = np.array([float(x_coords[idx2]), float(y_coords[idx2]), float(z_coords[idx2])])
                            dist = np.linalg.norm(p1 - p2)
                            rows_individual_distances.append([nom_fichier, chaine, f"{s1}-{s2}", round(dist, 4)])
                    
        except Exception as e:
            print(f"Erreur sur le fichier {filename} : {e}")

    # Écriture du CSV final (une ligne par distance)
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Fichier", "Chaine", "Intervalle_Residus", "Distance_A"])
        writer.writerows(rows_individual_distances)

    # Petit résumé dans la console
    if rows_individual_distances:
        toutes_les_dist = [r[3] for r in rows_individual_distances]
        print(f"\n--- ANALYSE TERMINÉE ---")
        print(f"Fichiers traités : {total_files}")
        print(f"Total de distances récoltées : {len(rows_individual_distances)}")
        print(f"Moyenne globale : {np.mean(toutes_les_dist):.4f} Å")
        print(f"Écart-type : {np.std(toutes_les_dist):.4f} Å")
        print(f"Fichier CSV créé : {output_path}")

# Lancement sur ton dossier
extract_individual_distances("dataset/1-5", "distances_individuelles_1-5.csv")