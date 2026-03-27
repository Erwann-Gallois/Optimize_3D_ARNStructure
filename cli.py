import argparse
import os
import sys
import time

# Add current folder and 'classe' folder to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classe"))

from fonction import read_fasta_file, generer_first_structure, pandaspdb_vers_cif
from classe.RigidEngine import RigidEngine
from classe.DFIREScoreModule import DFIREScoreModule
from classe.RASPScoreModule import RASPScoreModule

def main():
    parser = argparse.ArgumentParser(description="CLI interface for 3D RNA structure full-atom optimization.")
    
    # Argument pour sequence ou fasta file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sequence", type=str, help="RNA sequence directly")
    group.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file")
    
    # Argument pour scoring function
    parser.add_argument("--score", type=str, choices=["rasp", "dfire"], 
                        default="dfire", help="Full-atom scoring function to use (default: dfire)")
    
    # Arguments pour l'optimisation
    optimization_group = parser.add_argument_group("Optimization Parameters")
    optimization_group.add_argument("--epochs", type=int, default=50, help="Number of epochs per cycle (default: 50)")
    optimization_group.add_argument("--cycles", type=int, default=20, help="Number of cycles (default: 20)")
    optimization_group.add_argument("--lr", type=float, default=0.2, help="Learning rate (default: 0.2)")
    optimization_group.add_argument("--noise-coords", type=float, default=0.5, help="Noise on coordinates (default: 0.5)")
    optimization_group.add_argument("--noise-angles", type=float, default=0.2, help="Noise on angles (default: 0.2)") # Note: utilisé si implémenté dans l'Engine
    optimization_group.add_argument("--backbone-weight", type=float, default=100.0, help="Weight for backbone constraints (default: 100.0)")
    optimization_group.add_argument("--clash-weight", type=float, default=50.0, help="Weight for clash constraints (default: 50.0)")
    optimization_group.add_argument("--ref-atom", type=str, default="C3'", help="Reference atom for potential evaluation (default: C3')")

    # Arguments pour l'output et l'affichage
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument("-o", "--output", type=str, help="Name of the output file")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output during optimization")
    output_group.add_argument("--export_type", type=str, default="pdb", choices=["pdb", "cif"], help="Type of file to export")
    
    args = parser.parse_args()

    # 1. Récupération de la séquence
    sequence = ""
    if args.fasta:
        print(f"Reading FASTA file: {args.fasta}")
        try:
            from Bio import SeqIO
            record = next(SeqIO.parse(args.fasta, "fasta"))
            sequence = str(record.seq).upper().replace("T", "U")
        except Exception as e:
            print(f"Error reading FASTA: {e}")
            sys.exit(1)
    else:
        sequence = args.sequence.upper().replace("T", "U")

    print(f"Sequence loaded ({len(sequence)} nuc): {sequence[:50]}...")

    # 2. Génération de la structure initiale
    os.makedirs("fichier_arn", exist_ok=True)
    os.makedirs("resultat", exist_ok=True)
    
    initial_pdb = f"fichier_arn/initial_{int(time.time())}.pdb"
    print("Generating initial straight structure...")
    generer_first_structure(sequence, initial_pdb)
    
    if not os.path.exists(initial_pdb):
        print("Error: Initial structure could not be generated.")
        sys.exit(1)

    # 3. Préparation du chemin de sortie
    if args.output:
        output_path = f"resultat/{args.output}.{args.export_type}"
    else:
        output_path = f"resultat/optimized_{args.score}_{int(time.time())}.{args.export_type}"

    # 4. Initialisation du moteur Unifié (RigidEngine car on part d'un PDB)
    print(f"Initializing Unified Rigid Engine...")
    engine = RigidEngine(
        pdb_path=initial_pdb,
        ref_atom=args.ref_atom,
        backbone_weight=args.backbone_weight,
        clash_weight=args.clash_weight,
        verbose=args.verbose
    )

    # 5. Ajout de la fonction de score choisie (Composition)
    if args.score == "rasp":
        print("  -> Adding RASP scoring module")
        score_mod = RASPScoreModule(type_RASP="all", nbre_nt_exclu=2)
    else:
        print("  -> Adding DFIRE scoring module")
        # Assurez-vous que le chemin vers le potentiel est correct
        pot_path = "potentials/matrice_dfire.dat"
        score_mod = DFIREScoreModule(potential_path=pot_path, nbre_nt_exclu=2)
    
    engine.add_score_module(score_mod, weight=1.0)

    # 6. Lancement de l'optimisation
    print("\n--- Starting optimization ---")
    start_time = time.perf_counter()
    
    # run_optimization dans RigidEngine renvoie (PandasPdb_object, best_score)
    result_ppdb, best_score = engine.run_optimization(
        num_cycles=args.cycles,
        epochs=args.epochs,
        lr=args.lr,
        noise_coords=args.noise_coords
    )
    
    end_time = time.perf_counter()
    print("--- Optimization finished ---")

    # 7. Export selon le format demandé
    if args.export_type == "pdb":
        result_ppdb.to_pdb(output_path)
    elif args.export_type == "cif":
        pandaspdb_vers_cif(result_ppdb, output_path)
        
    print(f"\n✅ Best score obtained: {best_score:.4f}")
    print(f"✅ Result saved in: {output_path}")
    print(f"⏱ Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()