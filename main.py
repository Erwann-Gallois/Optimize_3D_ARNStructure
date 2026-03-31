import os
import argparse
from classe.BeadSpringRASPOptimizer import BeadSpringRASPOptimizer

def main():
    parser = argparse.ArgumentParser(description="Test de la classe BeadSpringRASPOptimizer")
    parser.add_argument("--pdb", type=str, default="fichier_arn/initial_1774627195.pdb", help="Chemin vers le fichier PDB d'entrée")
    parser.add_argument("--out", type=str, default="output_bead_optimized.pdb", help="Chemin vers le fichier PDB de sortie")
    parser.add_argument("--epochs", type=int, default=500, help="Nombre d'époques pour l'optimisation")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (taux d'apprentissage)")
    parser.add_argument("--rasp_weight", type=float, default=1.0, help="Poids de la fonction RASP dans la loss")
    parser.add_argument("--type_rasp", type=str, default="c3", help="Type de potentiel RASP à utiliser (ex: c3)")
    
    args = parser.parse_args()

    # Vérification de l'existence du fichier
    if not os.path.exists(args.pdb):
        print(f"Erreur : Le fichier {args.pdb} n'existe pas.")
        # Au cas où le fichier exact mentionné n'existe plus, on cherche initial.pdb
        fallback = "fichier_arn/initial.pdb"
        if os.path.exists(fallback):
            print(f"Utilisation du fichier de repli : {fallback}")
            args.pdb = fallback
        else:
            print("Veuillez spécifier un PDB valide avec l'argument --pdb.")
            return

    print(f"=== Optimisation Bead-Spring avec RASP ===")
    print(f"Fichier d'entrée : {args.pdb}")
    print(f"Fichier de sortie : {args.out}")
    print(f"Type RASP : {args.type_rasp} (Poids: {args.rasp_weight})")
    print(f"Époques : {args.epochs} | LR : {args.lr}")
    print("==========================================\n")

    # Instanciation de l'optimiseur avec nos paramètres de test
    optimizer = BeadSpringRASPOptimizer(
        pdb_path=args.pdb,
        lr=args.lr,
        output_path=args.out,
        num_epochs=args.epochs,
        bead_atom="C3'",
        k_fraenkel=40.0,
        r0_fraenkel=5.5,
        rmax_fene=1.5,
        epsilon_lj=0.2,
        sigma_lj=3.5,
        exclude_near_neighbors=2,
        angle_weight=2.0,
        target_angle_deg=120.0,
        type_RASP=args.type_rasp,
        rasp_weight=args.rasp_weight
    )

    # Lancement de la procédure d'optimisation
    optimizer.optimize()

if __name__ == "__main__":
    main()
