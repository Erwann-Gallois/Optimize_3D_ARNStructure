from fonction import generer_arn_droit, view_structure, enlever_hydrogene
from RNA_RASP_Gradient import RNA_RASP_Gradient

sequence_arn = "AUGCGAUUUAC"
generer_arn_droit(sequence_arn, "mon_arn_droit.pdb")
enlever_hydrogene("mon_arn_droit.pdb", "mon_arn_droit_sans_hydrogene.pdb")
RNA_RASP_Gradient("mon_arn_droit_sans_hydrogene.pdb", output_path="mon_arn_optimise.pdb").run_optimization()

