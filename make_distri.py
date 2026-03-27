import requests
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBList, is_aa
import os
import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="CLI interface for extracting atom distances from RNA structures.")
    parser.add_argument("-f", "--folder_path", help="Path to the folder containing CIF files.")
    parser.add_argument("-o", "--output_path", default="distances.csv", help="Path to the output CSV file.")
    parser.add_argument("--ref_atom_name", default="C3'", help="Name of the reference atom.")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the distribution of distances.")
    parser.add_argument("--export", action="store_true", help="Whether to export the distances to a CSV file.")
    args = parser.parse_args()
    extract_ref_atom_distance(args.folder_path, args.output_path, args.ref_atom_name, args.export, args.visualize)

def extract_ref_atom_distance(folder_path, file_path, ref_atom_name="C3'", export_csv=False, visualize=False):   
    parser = MMCIFParser(QUIET=True)
    distances = []
    sequence_lengths = []
    
    # Noms standards des nucléotides ARN dans les fichiers PDB/CIF
    rna_residues = ['A', 'C', 'G', 'U', 'RA', 'RC', 'RG', 'RU']

    for filename in os.listdir(folder_path):
        if filename.endswith(".cif"):
            structure_id = filename[:-4]
            try:
                structure = parser.get_structure(structure_id, os.path.join(folder_path, filename))
                
                for model in structure:
                    for chain in model:
                        # --- FILTRAGE ARN ---
                        # On ne garde que ce qui n'est pas une protéine ET qui est dans la liste ARN
                        rna_in_chain = [
                            res for res in chain.get_residues() 
                            if not is_aa(res) and res.get_resname().strip() in rna_residues
                        ]
                        
                        # Si la chaîne contient de l'ARN, on note sa longueur
                        if len(rna_in_chain) > 0:
                            sequence_lengths.append(len(rna_in_chain))
                        
                        # --- CALCUL DES DISTANCES C3' ---
                        # On parcourt la liste filtrée pour calculer les distances entre voisins
                        for i in range(len(rna_in_chain) - 1):
                            res1 = rna_in_chain[i]
                            res2 = rna_in_chain[i+1]
                            
                            # On vérifie que l'atome C3' est présent dans les deux nucléotides
                            if ref_atom_name in res1 and ref_atom_name in res2:
                                # Vérification de continuité (numéros de séquence consécutifs)
                                if res2.get_id()[1] == res1.get_id()[1] + 1:
                                    atom1 = res1[ref_atom_name]
                                    atom2 = res2[ref_atom_name]
                                    
                                    dist = atom1 - atom2 # Distance euclidienne
                                    distances.append(dist)
                                    
            except Exception as e:
                print(f"Erreur sur {filename}: {e}")

    if not distances:
        print("Aucune distance calculée. Vérifiez les fichiers CIF et le nom de l'atome de référence.")
        return
        
    df = pd.DataFrame(distances, columns=['Distance'])
    df_lengths = pd.DataFrame(sequence_lengths, columns=['Length'])
    
    if export_csv:
        df.to_csv(file_path, index=False)
        length_file_path = file_path.replace('.csv', '_lengths.csv')
        if length_file_path == file_path:
            length_file_path = file_path + "_lengths.csv"
        df_lengths.to_csv(length_file_path, index=False)
        print(f"Distances extraites et sauvegardées dans {file_path}.")
        print(f"Tailles extraites et sauvegardées dans {length_file_path}.")
    
    if visualize:
        visualise_distributions(df, df_lengths, ref_atom_name, export=export_csv)

    mean = df['Distance'].mean()
    std = df['Distance'].std()
    print(f"Distance moyenne : {mean:.3f} Å")
    print(f"Écart-type : {std:.3f} Å")
    
    return mean, std

def visualise_distributions(df_dist, df_lengths, ref_atom_name="C3'", export=False):
    data_dist = df_dist['Distance']
    data_len = df_lengths['Length']

    # 1. Calcul des paramètres statistiques Distances
    mu_dist = data_dist.mean()
    sigma_dist = data_dist.std()
    print(f"Distance Moyenne : {mu_dist:.3f} Å")
    print(f"Distance Écart-type : {sigma_dist:.3f} Å")

    # 2. Calcul des paramètres statistiques Tailles
    mu_len = data_len.mean()
    median_len = data_len.median()
    print(f"Taille moyenne : {mu_len:.1f} nucléotides")
    print(f"Taille médiane : {median_len:.1f} nucléotides")

    # 3. Création de la figure avec 2 sous-graphes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Sous-graphe 1 : Distances ---
    sns.histplot(data_dist, kde=True, color="skyblue", stat="density", label="Distances observées", edgecolor='black', ax=axes[0])
    x_dist = np.linspace(data_dist.min() - 0.5, data_dist.max() + 0.5, 100)
    p_dist = norm.pdf(x_dist, mu_dist, sigma_dist)
    axes[0].plot(x_dist, p_dist, 'r--', linewidth=2, label="Gausienne Théorique")
    
    axes[0].set_title(f"Distribution des distances {ref_atom_name}-{ref_atom_name} (n={len(data_dist)})", fontsize=14)
    axes[0].set_xlabel("Distance (Å)", fontsize=12)
    axes[0].set_ylabel("Densité de probabilité", fontsize=12)
    axes[0].axvline(mu_dist, color='red', linestyle='-', label=f'Moyenne = {mu_dist:.2f}Å')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # --- Sous-graphe 2 : Tailles ---
    sns.histplot(data_len, bins=30, color="lightgreen", edgecolor='black', ax=axes[1])
    
    axes[1].set_title(f"Distribution des tailles de séquences (n={len(data_len)})", fontsize=14)
    axes[1].set_xlabel("Taille de la séquence (nombre de nucléotides)", fontsize=12)
    axes[1].set_ylabel("Nombre de séquences", fontsize=12)
    axes[1].axvline(mu_len, color='red', linestyle='--', label=f'Moyenne = {mu_len:.1f}')
    axes[1].axvline(median_len, color='blue', linestyle='-', label=f'Médiane = {median_len:.1f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # 4. Sauvegarde
    if export:
        plt.savefig(f"distributions_combine_{ref_atom_name}.png", dpi=300)
    
    plt.show()

if __name__ == "__main__":
    main()



