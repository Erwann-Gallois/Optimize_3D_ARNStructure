from classe.RNA_DFIRE_C3_Optimizer import RNA_DFIRE_C3_Optimizer
from classe.RNA_RASP_C3_Optimizer import RNA_RASP_C3_Optimizer
import torch
import time
import os
from fonction import generer_arn_droit, read_fasta_file

execution_time = {}

start_time = time.perf_counter()
sequence_arn = "CCUGGUAUUGCAGUACCUCCAGGU"
pdb_initial = "fichier_arn/arn.pdb"
generer_arn_droit(sequence_arn, pdb_initial)
opt = RNA_RASP_C3_Optimizer(
    pdb_initial, 
    epochs_per_cycle=200,
    lr=0.2,
    output_path="resultat/arn_c3_optimisee_rasp" + ".pdb",
    num_cycles=150,
    backbone_weight=10.0
)

print("\n--- Démarrage de l'optimisation ---")
opt.run_optimization()
print("--- Fin de l'optimisation ---")
print("Affichage du score : ", opt.best_score)
end_time = time.perf_counter()
print(f"Optimisation terminée en {end_time - start_time:.4f} secondes.")