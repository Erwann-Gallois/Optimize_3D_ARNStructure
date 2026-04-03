import argparse
import os
import sys
import time

# Add current folder and 'classe' folder to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classe"))

from fonction import read_fasta_file, generer_first_structure

from BeadSpringRASPOptimizer import BeadSpringRASPOptimizer
from BeadSpringDFIREOptimizer import BeadSpringDFIREOptimizer

def main():
    parser = argparse.ArgumentParser(description="CLI interface for 3D RNA structure bead-spring optimization.")

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
    parser.add_argument("--noise-coords", type=float, default=0.5, help="Noise on coordinates (default: 10.0)")
    parser.add_argument("--k", type=float, default=40.0, help="Spring constant (default: 40.0)")
    parser.add_argument("--l0", type=float, default=5.5, help="Equilibrium length (default: 5.5)")
    parser.add_argument("--bead-atom", type=str, default="C3'", help="Atom to use as bead (default: C3')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during optimization")
    args = parser.parse_args()

    # 3. Optimizer selection
    output_path = args.output if args.output else f"resultat/optimized_bs_{args.score}_{int(time.time())}.pdb"
    
    # Security check: if output_path is a directory, append a default filename
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        filename = f"optimized_bs_{args.score}_{int(time.time())}.pdb"
        output_path = os.path.join(output_path, filename)
    elif not output_path.lower().endswith('.pdb'):
        output_path += ".pdb"

    optimizer_classes = {
        "rasp": BeadSpringRASPOptimizer,
        "dfire": BeadSpringDFIREOptimizer
    }
    
    OptClass = optimizer_classes[args.score]

    opt = OptClass(
            sequence = args.sequence,
            lr=args.lr,
            output_path=output_path,
            noise_coords=args.noise_coords,
            bead_atom=args.bead_atom,
            k=args.k,
            l0=args.l0,
            score_weight=1.0,
            verbose=args.verbose,
        )

    # 4. Optimization run
    print("\n--- Starting optimization ---")
    start_time = time.perf_counter()
    opt.run_optimization()
    end_time = time.perf_counter()
    print("--- Optimization finished ---")
    # print(f"Best score obtained: {opt.best_score}")
    print(f"Result saved in: {output_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()
