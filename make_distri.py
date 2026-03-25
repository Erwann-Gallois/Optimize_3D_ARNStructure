import requests
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBList
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

    args = parser.parse_args()
    extract_ref_atom_distance(args.folder_path, args.output_path, args.ref_atom_name)
    visualise_distribution(args.output_path, args.ref_atom_name)

def extract_ref_atom_distance(folder_path, file_path, ref_atom_name="C3'"):   
    parser = MMCIFParser(QUIET=True)
    distances = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".cif"):
            structure_id = filename[:-4]
            try:
                structure = parser.get_structure(structure_id, os.path.join(folder_path, filename))
                for model in structure:
                    for chain in model:
                        # On récupère tous les résidus de la chaîne
                        residues = list(chain.get_residues())
                        
                        for i in range(len(residues) - 1):
                            res1 = residues[i]
                            res2 = residues[i+1]
                            
                            # On vérifie que l'atome cible existe dans les deux résidus qui se suivent
                            if ref_atom_name in res1 and ref_atom_name in res2:
                                atom1 = res1[ref_atom_name]
                                atom2 = res2[ref_atom_name]
                                
                                # Vérification de continuité (pas de saut de numéro dans la séquence)
                                # res1.get_id()[1] donne le numéro du nucléotide
                                if res2.get_id()[1] == res1.get_id()[1] + 1:
                                    distance = atom1 - atom2  # L'opérateur '-' calcule la distance euclidienne
                                    distances.append(distance)
            except Exception as e:
                print(f"Erreur sur {filename}: {e}")

    if not distances:
        print("Aucune distance calculée. Vérifiez les fichiers CIF et le nom de l'atome de référence.")
        return
        
    df = pd.DataFrame(distances, columns=['Distance'])
    df.to_csv(file_path, index=False)
    print(f"Extraction terminée : {len(distances)} distances consécutives sauvegardées.")

def visualise_distribution(file_path, ref_atom_name="C3'"):
    df = pd.read_csv(file_path) 
    data = df['Distance']

    # 2. Calcul des paramètres statistiques
    mu = data.mean()
    sigma = data.std()
    print(f"Moyenne : {mu:.3f} Å")
    print(f"Écart-type : {sigma:.3f} Å")

    # 3. Création de la figure
    plt.figure(figsize=(10, 6))

    # Histogramme + Courbe de densité réelle (KDE)
    sns.histplot(data, kde=True, color="skyblue", stat="density", label="Distances observées", edgecolor='black')

    # Superposition de la Gaussienne Théorique (ton futur potentiel)
    x = np.linspace(data.min() - 0.5, data.max() + 0.5, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'r--', linewidth=2, label="Gausienne Théorique")

    # Personnalisation
    plt.title(f"Distribution des distances {ref_atom_name}-{ref_atom_name} (n={len(data)})", fontsize=14)
    plt.xlabel("Distance (Å)", fontsize=12)
    plt.ylabel("Densité de probabilité", fontsize=12)
    plt.axvline(mu, color='red', linestyle='-', label=f'Moyenne = {mu:.2f}Å')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 4. Sauvegarde
    plt.savefig(f"distribution_{ref_atom_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()



