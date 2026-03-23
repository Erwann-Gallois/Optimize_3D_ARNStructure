from classe.RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer
from fonction import generer_first_structure

sequence = "GUCUACCUAUCGGGCUAAGGAGCCGUAUGCGAUGAAAGUCGCACGUACGGUUCUAUGCCCGGGGGAAAAC"
generer_first_structure(sequence, "fichier_arn/initial.pdb")

opt = RNA_DFIRE_Optimizer(
    pdb_path = "fichier_arn/initial.pdb",
    ref_atom = "C3'",
    output_path = "fichier_arn/initial_optimized.pdb",
    epochs_per_cycle=100,
    lr=0.2,
    num_cycles=5,
    noise_coords=1.5,
    noise_angles=0.5,
    backbone_weight=int(len(sequence) * 2),
    # verbose=True
)

opt.run_optimization()
