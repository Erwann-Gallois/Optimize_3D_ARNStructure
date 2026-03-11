import torch
from fonction import generer_arn_droit
from classe.RNA_RASP_Rigid import RNA_RASP_Rigid

# Force PyTorch à utiliser 50 threads sur le CPU
torch.set_num_threads(50)

# 1. Génération de la structure initiale (ARN droit)
sequence_arn = "GGAACCGGUGCGCAUAACCACCUCAGUGCGAGCAA"
pdb_initial = "fichier_arn/mon_arn_rigid_droit.pdb"
generer_arn_droit(sequence_arn, pdb_initial)

# 2. Optimisation par corps rigide (Nucleotide par Nucleotide)
# On utilise la nouvelle classe RNA_RASP_Rigid
# ref_atom="C3'" par défaut
opt = RNA_RASP_Rigid(
    pdb_initial, 
    epochs_per_cycle=100,
    lr=0.1,
    type_RASP="all",
    output_path="mon_arn_rigide_optimise.pdb",
    ref_atom="C3'",
    num_cycles=10,
    noise_coords=2,
    noise_angles=1
)

print("\n--- Démarrage de l'optimisation rigide ---")
opt.run_optimization()
print("--- Fin de l'optimisation ---")
