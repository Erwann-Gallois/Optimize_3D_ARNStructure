import torch
from fonction import generer_arn_droit
from classe.RNA_RASP_Rigid import RNA_RASP_Rigid
from classe.RNA_RASP_C3_Gradient import RNA_RASP_C3_Annealing

# 1. Génération de la structure initiale (ARN droit)
sequence_arn = "GGAACCGGUGCGCAUAACCACCUCAGUGCGAGCAA"
pdb_initial = "fichier_arn/mon_arn_rigid_droit.pdb"
generer_arn_droit(sequence_arn, pdb_initial)

# # 2. Optimisation par corps rigide (Nucleotide par Nucleotide)
# # On utilise la nouvelle classe RNA_RASP_Rigid
# # ref_atom="C3'" par défaut
# opt = RNA_RASP_Rigid(
#     pdb_initial, 
#     epochs_per_cycle=800,
#     lr=0.1,
#     type_RASP="all",
#     output_path="mon_arn_rigide_optimise.pdb",
#     ref_atom="C3'",
#     num_cycles=100,
#     noise_coords=2.0,
#     noise_angles=0.8
# )

# print("\n--- Démarrage de l'optimisation rigide ---")
# opt.run_optimization()
# print("--- Fin de l'optimisation ---")

opt = RNA_RASP_C3_Annealing(pdb_initial, "potentials/c3.nrg", "mon_arn_c3_optimise.pdb")
opt.run_annealing()
