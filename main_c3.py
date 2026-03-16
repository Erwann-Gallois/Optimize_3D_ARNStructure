from classe.RNA_C3_Optimizer import RNA_C3_Optimizer
import torch
import time
import os
from fonction import generer_arn_droit, read_fasta_file

execution_time = {}
seqs, nom = read_fasta_file("example.fasta")
# 1. Génération de la structure initiale (ARN droit)
for seq, name in zip(seqs, nom):
    start_time = time.perf_counter()
    taille = len(seq)
    pdb_initial = "fichier_arn/arn_" + str(taille) + ".pdb"
    generer_arn_droit(seq, pdb_initial)
    opt = RNA_C3_Optimizer(
        pdb_initial, 
        epochs_per_cycle=100,
        lr=0.2,
        output_path="resultat/arn_c3_optimisee_" + str(taille) + ".pdb",
        num_cycles=100,
        backbone_weight=100.0
    )

    print("\n--- Démarrage de l'optimisation ---")
    opt.run_optimization()
    print("--- Fin de l'optimisation ---")
    print("Affichage du score : ", opt.best_score)
    end_time = time.perf_counter()
    name = name + "-" + str(taille)
    execution_time[name] = end_time - start_time
    print(f"Optimisation terminée en {execution_time[name]:.4f} secondes.")

with open("execution_time_c3.txt", "w") as f:
    for name, time in execution_time.items():
        f.write(f"{name}: {time:.4f}\n")