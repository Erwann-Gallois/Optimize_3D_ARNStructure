import torch
from fonction import generer_arn_droit, read_fasta_file
from classe.RNA_RASP_Optimizer import RNA_RASP_Optimizer
from classe.RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer
import time
import os

# 1. Génération de la structure initiale (ARN droit)

start_time = time.perf_counter()
sequence_arn = "CCUGGUAUUGCAGUACCUCCAGGU"
pdb_initial = "fichier_arn/arn.pdb"
generer_arn_droit(sequence_arn, pdb_initial)
opt = RNA_RASP_Optimizer(
    pdb_initial,
    epochs_per_cycle=100,
    lr=0.1,
    output_path="resultat/mon_arn_optimise_rasp.pdb",
    ref_atom="C3'",
    num_cycles=100,
    noise_coords=10.0,
    noise_angles=15.0,
    backbone_weight=100.0
)

print("\n--- Démarrage de l'optimisation ---")
opt.run_optimization()
print("--- Fin de l'optimisation ---")
print("Affichage du score : ", opt.best_score)
end_time = time.perf_counter()
print("Réalisé en :", end_time - start_time)