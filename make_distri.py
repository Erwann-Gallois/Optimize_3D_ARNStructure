import numpy as np
import os
import csv
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def extract_individual_distances(folder_path, output_path, ref_atom="C3'"):
    # Liste pour stocker chaque mesure individuelle
    # Format : [Nom_Fichier, Chaine, Index_Depart, Distance]
    dist_data = []
    angle_data = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".cif"): continue
        try:
            data_dict = MMCIF2Dict(os.path.join(folder_path, filename))
            
            # Extraction des colonnes nécessaires
            atom_names = data_dict["_atom_site.label_atom_id"]
            asym_ids = data_dict["_atom_site.label_asym_id"]
            seq_ids = data_dict["_atom_site.label_seq_id"]
            x_coords = np.array(data_dict["_atom_site.Cartn_x"], dtype=float)
            y_coords = np.array(data_dict["_atom_site.Cartn_y"], dtype=float)
            z_coords = np.array(data_dict["_atom_site.Cartn_z"], dtype=float)
            
            coords_all = np.stack([x_coords, y_coords, z_coords], axis=1)
            chains = sorted(list(set(asym_ids)))

            for chaine in chains:
                # Indices des atomes cibles dans la chaîne
                idx_list = [i for i, name in enumerate(atom_names) if name == ref_atom and asym_ids[i] == chaine]
                
                for j in range(len(idx_list)):
                    # 1. DISTANCE (entre j et j+1)
                    if j < len(idx_list) - 1:
                        i1, i2 = idx_list[j], idx_list[j+1]
                        if int(seq_ids[i2]) == int(seq_ids[i1]) + 1:
                            d = np.linalg.norm(coords_all[i2] - coords_all[i1])
                            dist_data.append([filename, chaine, f"{seq_ids[i1]}-{seq_ids[i2]}", round(d, 4)])
                    
                    # 2. ANGLE (entre j, j+1 et j+2)
                    if j < len(idx_list) - 2:
                        i1, i2, i3 = idx_list[j], idx_list[j+1], idx_list[j+2]
                        if int(seq_ids[i2]) == int(seq_ids[i1]) + 1 and int(seq_ids[i3]) == int(seq_ids[i2]) + 1:
                            v1 = coords_all[i1] - coords_all[i2]
                            v2 = coords_all[i3] - coords_all[i2]
                            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                            angle_data.append([filename, chaine, f"{seq_ids[i1]}-{seq_ids[i2]}-{seq_ids[i3]}", round(angle_deg, 4)])

        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    # Sauvegarde des deux fichiers
    for path, data, head in [(output_path + "_dist.csv", dist_data, ["Fichier", "Chaine", "Residus", "Distance_A"]), 
                             (output_path + "_angle.csv", angle_data, ["Fichier", "Chaine", "Residus", "Angle_Deg"])]:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            writer.writerows(data)

    # Petit résumé dans la console
    if dist_data:
        toutes_les_dist = [r[3] for r in dist_data]
        print(f"\n--- ANALYSE TERMINÉE ---")
        print(f"Fichiers traités : {total_files}")
        print(f"Total de distances récoltées : {len(rows_individual_distances)}")
        print(f"Moyenne globale : {np.mean(toutes_les_dist):.4f} Å")
        print(f"Écart-type : {np.std(toutes_les_dist):.4f} Å")
        print(f"Fichier CSV créé : {output_path}")

# Lancement sur ton dossier
extract_individual_distances("dataset/0-1", "distances_individuelles_0-1.csv")
