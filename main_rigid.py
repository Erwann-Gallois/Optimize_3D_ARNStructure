import torch
from fonction import generer_arn_droit
from classe.RNA_RASP_Rigid import RNA_RASP_Rigid

# 1. Génération de la structure initiale (ARN droit)
sequence_arn = "CCUGGUAUUGCAGUACCUCCAGGU"
pdb_initial = "fichier_arn/mon_arn_rigid_droit.pdb"
generer_arn_droit(sequence_arn, pdb_initial)

# # 2. Optimisation par corps rigide (Nucleotide par Nucleotide)
# # On utilise la nouvelle classe RNA_RASP_Rigid
# # ref_atom="C3'" par défaut
opt = RNA_RASP_Rigid(
    pdb_initial, 
    epochs_per_cycle=150,
    lr=0.1,
    type_RASP="all",
    output_path="resultat/mon_arn_rigide_optimise_1.pdb",
    ref_atom="C3'",
    num_cycles=50,
    noise_coords=5.0,
    noise_angles=5.0,
    backbone_weight=100.0
)

print("\n--- Démarrage de l'optimisation rigide ---")
opt.run_optimization()
print("--- Fin de l'optimisation ---")
