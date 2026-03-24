from classe.RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer
from classe.RNA_RASP_Optimizer import RNA_RASP_Optimizer
from fonction import generer_first_structure

sequence = "GACACUAAGUUCGGCAUCAAUAUGGUGACCUCCCGGGAGCGGGGGACCACCAGGUUGCCUAAGGAGGGGUGAACCGGCCCAGGUCGGAAACGGAGCAGGUCAAAACUCCCGUGCUGAUCAGUAGUGU"
generer_first_structure(sequence, "fichier_arn/initial.pdb")

opt = RNA_RASP_Optimizer(
    pdb_path = "fichier_arn/initial.pdb",
    ref_atom = "C3'",
    output_path = "fichier_arn/initial_optimized.pdb",
    lr=0.1,
    num_cycles=50,
    epochs_per_cycle=100,
    noise_coords=10.0,
    noise_angles=15.0,
    backbone_weight=100.0
    # verbose=True
)

opt.run_optimization()
