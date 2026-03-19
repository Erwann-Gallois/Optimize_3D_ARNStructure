import torch
from fonction import generer_arn_droit, read_fasta_file, enlever_hydrogene
from classe.RNA_RASP_Optimizer import RNA_RASP_Optimizer
from classe.RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer
import time
import os

# 1. Génération de la structure initiale (ARN droit)

start_time = time.perf_counter()
pdb_initial = "fichier_arn/arn_24_sans_h.pdb"
opt = RNA_DFIRE_Optimizer(
    pdb_initial,
    epochs_per_cycle=50,
    lr=0.2,
    output_path="resultat/mon_arn_optimise_dfire.pdb",
    ref_atom="C3'",
    num_cycles=20,
    noise_coords=10.0,
    noise_angles=15.0,
    backbone_weight=100.0,
    verbose=False
)

print("\n--- Démarrage de l'optimisation ---")
# opt.calculate_detailed_scores = torch.compile(opt.calculate_detailed_scores)
opt.run_optimization()
print("--- Fin de l'optimisation ---")
print("Affichage du score : ", opt.best_score)
end_time = time.perf_counter()
print("Réalisé en :", end_time - start_time)