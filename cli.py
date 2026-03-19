import argparse
import os
import sys
import time

# Add current folder and 'classe' folder to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classe"))

from fonction import read_fasta_file, generer_arn_droit, enlever_hydrogene, fix_amber_pdb
from RNA_RASP_Optimizer import RNA_RASP_Optimizer
from RNA_DFIRE_Optimizer import RNA_DFIRE_Optimizer

def main():
    parser = argparse.ArgumentParser(description="CLI interface for 3D RNA structure full-atom optimization.")
    
    # Argument for sequence or fasta file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sequence", type=str, help="RNA sequence directly")
    group.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file")
    
    # Argument for scoring function (restricted to full atoms for now)
    parser.add_argument("--score", type=str, choices=["rasp", "dfire"], 
                        default="dfire", help="Full-atom scoring function to use (default: dfire)")
    
    # Arguments for optimization
    parser.add_argument("-o", "--output", type=str, help="Output PDB file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per cycle (default: 50)")
    parser.add_argument("--cycles", type=int, default=20, help="Number of cycles (default: 20)")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate (default: 0.2)")
    parser.add_argument("--noise-coords", type=float, default=10.0, help="Noise on coordinates (default: 10.0)")
    parser.add_argument("--noise-angles", type=float, default=15.0, help="Noise on angles (default: 15.0)")
    parser.add_argument("--backbone-weight", type=float, default=100.0, help="Backbone weight (default: 100.0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()

    # 1. Sequence retrieval
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

    # 2. Initial structure generation
    os.makedirs("fichier_arn", exist_ok=True)
    os.makedirs("resultat", exist_ok=True)
    
    initial_pdb = f"fichier_arn/initial_{int(time.time())}.pdb"
    initial_pdb_fix = f"fichier_arn/initial_{int(time.time())}_fixed.pdb"
    initial_pdb_no_h = initial_pdb.replace(".pdb", "_no_h.pdb")
    
    print("Generating initial straight structure...")
    generer_arn_droit(sequence, initial_pdb)
    fix_amber_pdb(initial_pdb, initial_pdb_fix)  # Fix any potential issues with the generated PDB
    
    if not os.path.exists(initial_pdb_fix):
        print("Error: Initial structure could not be generated.")
        sys.exit(1)
        
    print("Removing hydrogens...")
    enlever_hydrogene(initial_pdb_fix, initial_pdb_no_h)


    # 3. Optimizer selection
    output_path = args.output if args.output else f"resultat/optimized_{args.score}_{int(time.time())}.pdb"
    
    # Security check: if output_path is a directory, append a default filename
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        filename = f"optimized_{args.score}_{int(time.time())}.pdb"
        output_path = os.path.join(output_path, filename)
    elif not output_path.lower().endswith('.pdb'):
        output_path += ".pdb"

    optimizer_classes = {
        "rasp": RNA_RASP_Optimizer,
        "dfire": RNA_DFIRE_Optimizer
    }
    
    OptClass = optimizer_classes[args.score]
    
    print(f"Initializing {args.score} optimizer (full atoms)...")
    
    # Unified call for full-atom optimizers
    # Passing everything as named arguments for Clarity
    opt = OptClass(
        pdb_path=initial_pdb_no_h,
        epochs_per_cycle=args.epochs,
        lr=args.lr,
        output_path=output_path,
        ref_atom="all", # Force all for full atom
        num_cycles=args.cycles,
        noise_coords=args.noise_coords,
        noise_angles=args.noise_angles,
        backbone_weight=args.backbone_weight,
        verbose=args.verbose
    )

    # 4. Optimization run
    print("\n--- Starting optimization ---")
    start_time = time.perf_counter()
    opt.run_optimization()
    end_time = time.perf_counter()
    print("--- Optimization finished ---")
    
    print(f"Best score obtained: {opt.best_score}")
    print(f"Result saved in: {output_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
