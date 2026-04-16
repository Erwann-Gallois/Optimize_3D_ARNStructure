import argparse
import os
import sys
import time
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add current folder and 'classe' folder to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classe"))

from fonction import read_fasta_file, generer_first_structure
from FullAtomRASPOptimizer import FullAtomRASPOptimizer
from FullAtomDFIREOptimizer import FullAtomDFIREOptimizer
from FullAtomRsRNASPOptimizer import FullAtomRsRNASPOptimizer

def display_summary(args, output_path):
    console = Console()
    
    # Création du tableau
    table = Table(title="Résumé des Paramètres d'Optimisation", title_style="bold magenta")

    # Définition des colonnes
    table.add_column("Paramètre", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="green")

    table.add_row("Séquence", args.sequence if args.sequence else args.fasta),
    table.add_row("Fonction de Score", args.score.upper()),
    table.add_row("Sortie", output_path),
    table.add_row("Patience (Local/Globale)", f"{args.patience_locale} / {args.patience_globale}"),
    table.add_row("Min Delta", str(args.min_delta)),
    table.add_row("Taux de Refroidissement", str(args.taux_refroidissement)),
    table.add_row("Bruit minimum", f"{args.bruit_min} Å"),
    table.add_row("Bruit initial", f"{args.noise_coords} Å"),
    table.add_row("Bruit sur les angles", f"{args.noise_angles} rad"),
    table.add_row("Backbone Weight", str(args.backbone_weight)),
    # Affichage
    console.print(Panel(table, expand=False, border_style="magenta"))


def main():
    parser = argparse.ArgumentParser(description="CLI interface for 3D RNA structure full-atom optimization.")
    
    # Argument for sequence or fasta file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sequence", type=str, help="RNA sequence directly")
    group.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file")
    
    # Argument for scoring function (restricted to full atoms for now)
    parser.add_argument("--score", type=str, choices=["rasp", "dfire", "rsRNASP"], 
                        default="rasp", help="Full-atom scoring function to use (default: dfire)")
    
    # Arguments for optimization
    parser.add_argument("-o", "--output", type=str, help="Output PDB file")
    parser.add_argument("--patience-locale", type=int, default=100, help="Iterations without improvement before stopping local phase (default: 100)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum energy change to be considered an improvement (default: 1e-4)")
    parser.add_argument("--patience-globale", type=int, default=5, help="Shakes without finding a new record before stopping (default: 5)")
    parser.add_argument("--taux-refroidissement", type=float, default=0.85, help="Reduction factor for noise after each shake (default: 0.85)")
    parser.add_argument("--bruit-min", type=float, default=0.01, help="Noise threshold to stop optimization (default: 0.01)")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate (default: 0.2)")
    parser.add_argument("--backbone-weight", type=int, default=100, help="Weight of the backbone atoms (default: 100)")
    parser.add_argument("--noise-coords", type=float, default=0.5, help="Noise on coordinates (default: 10.0)")
    parser.add_argument("--noise-angles", type=float, default=0.2, help="Noise on angles (default: 15.0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during optimization")
    parser.add_argument("--cif", action="store_true", help="Export the final structure in .cif format")
    parser.add_argument("--confirm", action="store_true", help="Confirm the optimization parameters before running")
    args = parser.parse_args()

    # 1. Sequence retrieval
    sequence = ""
    if args.fasta:
        print(f"Reading FASTA file: {args.fasta}")
        click.secho(f"Reading FASTA file... : {args.fasta}", fg='cyan')
        try:
            from Bio import SeqIO
            record = next(SeqIO.parse(args.fasta, "fasta"))
            sequence = str(record.seq).upper().replace("T", "U")
        except Exception as e:
            click.secho(f"Error reading FASTA: {e}", fg='red')
            sys.exit(1)
    else:
        sequence = args.sequence.upper().replace("T", "U")

    click.secho(f"Sequence loaded ({len(sequence)} nuc): {sequence[:50]}...", fg='cyan')

    # 2. Initial structure generation
    os.makedirs("fichier_arn", exist_ok=True)
    os.makedirs("resultat", exist_ok=True)
    
    initial_pdb = f"fichier_arn/initial_{int(time.time())}.pdb"
    generer_first_structure(sequence, initial_pdb)
    
    click.secho("Generating initial straight structure...", fg='cyan')
    
    
    if not os.path.exists(initial_pdb):
        click.secho("Error: Initial structure could not be generated.", fg='red')
        sys.exit(1)
    # 3. Optimizer selection
    output_path = args.output if args.output else f"resultat/optimized_fa_{args.score}_{int(time.time())}.pdb"
    
    # Security check: if output_path is a directory, append a default filename
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        filename = f"optimized_fa_{args.score}_{int(time.time())}.pdb"
        output_path = os.path.join(output_path, filename)
    elif not output_path.lower().endswith('.pdb'):
        output_path += ".pdb"

    if args.confirm:
        # --- ÉTAPE DE VALIDATION AJOUTÉE ---
        display_summary(args, output_path)

        # Demande de confirmation
        if not click.confirm(click.style("\n🚀 Souhaitez-vous lancer l'optimisation avec ces paramètres ?", fg='magenta', bold=True), default=True):
            click.secho("❌ Opération annulée par l'utilisateur.", fg='red')
            sys.exit(0)

    optimizer_classes = {
        "rasp": FullAtomRASPOptimizer,
        "dfire": FullAtomDFIREOptimizer,
        "rsRNASP": FullAtomRsRNASPOptimizer,
    }
    
    OptClass = optimizer_classes[args.score]
    
    click.secho(f"Initializing {args.score} optimizer (full atoms)...", fg='cyan')
    
    # Unified call for full-atom optimizers
    # Passing everything as named arguments for Clarity
    opt = OptClass(
        pdb_path=initial_pdb,
        lr=args.lr,
        output_path=output_path,
        ref_atom="all", # Force all for full atom
        noise_coords=args.noise_coords,
        noise_angles=args.noise_angles,
        backbone_weight=args.backbone_weight,
        verbose=args.verbose,
        patience_locale=args.patience_locale,
        min_delta=args.min_delta,
        patience_globale=args.patience_globale,
        taux_refroidissement=args.taux_refroidissement,
        bruit_min=args.bruit_min,
        export_cif=args.cif
    )

    # 4. Optimization run
    start_time = time.perf_counter()
    opt.run_optimization()
    end_time = time.perf_counter()
    click.secho(f"Best score obtained: {opt.best_score}", fg='green')
    click.secho(f"Result saved in: {output_path}", fg='green')
    click.secho(f"Execution time: {end_time - start_time:.2f} seconds", fg='green', bold=True)

if __name__ == "__main__":
    main()