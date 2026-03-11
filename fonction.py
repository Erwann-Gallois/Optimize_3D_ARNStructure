import os
import subprocess
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt 
from biopandas.pdb import PandasPdb
import nglview as nv

def view_structure(file_pdb):
    # 1. Charger le fichier généré
    view = nv.show_file(file_pdb)
    # 2. Vider les représentations par défaut
    view.clear_representations()
    # 3. Afficher toute la molécule de manière fine
    view.add_representation('licorice', selection='all', color='element')
    # 4. Mettre en surbrillance spéciale les atomes de Phosphore (les "noeuds" des liaisons phosphodiester)
    view.add_representation('spacefill', selection='_P', color='orange', radius=0.8)
    # 5. Rajouter un "tube" qui suit et met en évidence de façon continue le squelette phosphodiester (backbone)
    view.add_representation('tube', selection='backbone', color='red', radius=0.2)
    # 6. Centrer la vue
    view.center()
    # Afficher l'interface
    return view

def enlever_hydrogene(file_pdb, output_file_pdb):
    ppdb_df =  PandasPdb().read_pdb(file_pdb)
    atom_df = ppdb_df.df["ATOM"]
    atom_sans_h_df = atom_df[~atom_df["atom_name"].str.startswith("H")]
    atom_sans_h_df.head(20)
    ppdb_df.df['ATOM'] = atom_sans_h_df
    ppdb_df.to_pdb(output_file_pdb)

def generer_arn_droit(sequence, fichier_sortie="arn_structure.pdb"):
    """
    Génère un fichier PDB d'un ARN simple brin droit (canonique) à partir 
    d'une séquence de nucléotides.
    Nécessite AmberTools ('tleap' installé et dans le PATH).
    """
    # 1. Formater la séquence pour tleap
    # Pour éviter les erreurs de syntaxe avec des séquences très longues,
    # on découpe la séquence en morceaux (chunks) de 50 nucléotides.
    taille_chunk = 50
    chunks = [sequence[i:i + taille_chunk] for i in range(0, len(sequence), taille_chunk)]
    
    script_tleap = "instructions_tleap.in"
    
    # 2. Créer le script d'instructions que l'on passera à tleap
    with open(script_tleap, "w") as f:
        # Charger le champ de force pour l'ARN (OL3 est le standard très recommandé)
        f.write("source leaprc.RNA.OL3\n")
        
        # Construire chaque morceau de la séquence
        noms_parties = []
        for i, chunk in enumerate(chunks):
            chunk_formatee = " ".join([nuc.upper() for nuc in chunk])
            nom_partie = f"partie_{i}"
            f.write(f"{nom_partie} = sequence {{ {chunk_formatee} }}\n")
            noms_parties.append(nom_partie)
        
        # Combiner tous les morceaux dans une variable "mon_arn"
        liste_parties = " ".join(noms_parties)
        f.write(f"mon_arn = combine {{ {liste_parties} }}\n")
        
        # Exporter la molécule au format PDB
        f.write(f"savepdb mon_arn {fichier_sortie}\n")
        
        # Quitter le programme
        f.write("quit\n")
        
    print(f"Génération de la structure pour la séquence ({len(sequence)} nuc) : {sequence[:50]}...")
    
    # 3. Lancer tleap depuis Python en tant que sous-processus
    try:
        # Note: tleap doit être dans votre PATH (généralement après conda activate Stage)
        process = subprocess.run(
            ["tleap", "-f", script_tleap], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True,
            text=True
        )
        print(f"✅ Succès ! Le fichier '{fichier_sortie}' a été créé.")
        
    except FileNotFoundError:
        print("❌ Erreur : 'tleap' est introuvable.")
        print("Assure-toi d'avoir activé ton environnement avec conda : conda activate Stage")
    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'exécution de tleap !")
        print("Détails stdout :", e.stdout)
        print("Détails stderr :", e.stderr)
        
    finally:
        # 4. Nettoyer les fichiers temporaires générés par AmberTools
        if os.path.exists(script_tleap):
            os.remove(script_tleap)
        if os.path.exists("leap.log"):
            os.remove("leap.log")

def read_fasta_file (fasta_file):
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"Le fichier {fasta_file} n'existe pas.")
    if not os.path.endswith(".fasta"):
        raise ValueError(f"Le fichier {fasta_file} n'est pas un fichier fasta.")
    sequences = []
    records = list(SeqIO.parse(fasta_file, "fasta"))
    for record in records:
        sequences.append(record.seq)
    return sequences
        