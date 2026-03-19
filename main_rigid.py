import torch
from fonction import generer_arn_droit, read_fasta_file
from classe.RNA_RASP_Optimizer import RNA_RASP_Optimizer
from classe.RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer
import time
import os

execution_time = {}
seqs, nom = read_fasta_file("example.fasta")
# 1. Génération de la structure initiale (ARN droit)


for seq, name in zip(seqs, nom):
    start_time = time.perf_counter()
    taille = len(seq)
    pdb_initial = "fichier_arn/arn_" + str(taille) + ".pdb"
    opt = RNA_DFIRE_Optimizer(
        pdb_initial, 
       epochs_per_cycle=50,
        lr=0.2,
        output_path="resultat/mon_arn_optimise_" + str(taille) + "_dfire.pdb",
        ref_atom="C3'",
        num_cycles=20,
        noise_coords=10.0,
        noise_angles=15.0,
        backbone_weight=100.0,
        verbose=False
    )

    print("\n--- Démarrage de l'optimisation ---")
    opt.run_optimization()
    print("--- Fin de l'optimisation ---")
    print("Affichage du score : ", opt.best_score)
    end_time = time.perf_counter()
    name = name + "-" + str(taille)
    execution_time[name] = end_time - start_time
    print(f"Optimisation terminée en {execution_time[name]:.4f} secondes.")

with open("execution_time_dfire_cpu.txt", "w") as f:
    for name, time in execution_time.items():
        f.write(f"{name}: {time:.4f}\n")