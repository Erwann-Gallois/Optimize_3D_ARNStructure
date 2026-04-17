import argparse
import os
import sys
import time
import pandas as pd
import click

# Add current folder and 'classe' folder to PYTHONPATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classe"))

from fonction import read_fasta_file, generer_first_structure

from BeadSpringRASPOptimizer import BeadSpringRASPOptimizer
from BeadSpringDFIREOptimizer import BeadSpringDFIREOptimizer
from BeadSpringRsRNASPOptimizer import BeadSpringRsRNASPOptimizer

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def display_summary(args, output_path):
    console = Console()
    
    # Création du tableau
    table = Table(title="Résumé des Paramètres d'Optimisation", title_style="bold magenta")

    # Définition des colonnes
    table.add_column("Paramètre", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="green")

    table.add_row("Séquence", str(args.sequence if args.sequence else args.fasta))
    table.add_row("Fonction de Score", str(args.score.upper()))
    table.add_row("Sortie", str(output_path))
    table.add_row("Patience (Local/Globale)", str(f"{args.patience_locale} / {args.patience_globale}"))
    table.add_row("Min Delta", str(args.min_delta))
    table.add_row("Taux de Refroidissement", str(args.taux_refroidissement))
    table.add_row("Bruit minimum", str(f"{args.bruit_min} Å"))
    table.add_row("Bruit initial", str(f"{args.noise_coords} Å"))
    table.add_row("Constante K", str(args.k))
    table.add_row("Poids du Score", str(args.score_weight))
    # Affichage
    console.print(Panel(table, title="[bold cyan]Configuration de l'Optimisation[/]", expand=False))


def main():
    parser = argparse.ArgumentParser(description="CLI interface for 3D RNA structure bead-spring optimization.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sequence", type=str, help="RNA sequence directly")
    group.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file")

    # Argument for scoring function (restricted to full atoms for now)
    parser.add_argument("--score", type=str, choices=["rasp", "dfire", "rsRNASP"], 
                        default="dfire", help="Full-atom scoring function to use (default: dfire)")
    
    # Arguments for optimization
    parser.add_argument("-o", "--output", type=str, help="Output PDB file")
    parser.add_argument("--patience-locale", type=int, default=100, help="Number of epochs before local optimization (default: 100)")
    parser.add_argument("--patience-globale", type=int, default=5, help="Number of epochs before global optimization (default: 5)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum delta for local optimization (default: 1e-4)")
    parser.add_argument("--taux-refroidissement", type=float, default=0.85, help="Rate of temperature decrease (default: 0.85)")
    parser.add_argument("--bruit-min", type=float, default=0.01, help="Minimum temperature (default: 0.01)")
    parser.add_argument("--noise-coords", type=float, default=0.5, help="Noise on coordinates (default: 10.0)")
    parser.add_argument("--k", type=float, default=40.0, help="Spring constant (default: 40.0)")
    parser.add_argument("--l0", type=float, default=5.5, help="Equilibrium length (default: 5.5)")
    parser.add_argument("--bead-atom", type=str, default="C3'", help="Atom to use as bead (default: C3')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output during optimization")
    parser.add_argument("--cif", action="store_true", help="Export the final structure in .cif format")
    parser.add_argument("--score_weight", type=float, default=1.0, help="Weight for the scoring function (default: 1.0)")
    parser.add_argument("--confirm", action="store_true", help="Confirm the optimization parameters before running")
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

    if args.confirm:
        # --- ÉTAPE DE VALIDATION AJOUTÉE ---
        display_summary(args, output_path)

        # Demande de confirmation
        if not click.confirm(click.style("\n Souhaitez-vous lancer l'optimisation avec ces paramètres ?", fg='magenta', bold=True), default=True):
            click.secho("Opération annulée par l'utilisateur.", fg='red')
            sys.exit(0)
    dist_csv = pd.read_csv("results.csv")
    result = dist_csv[dist_csv["Ref Atom"] == args.bead_atom]
    if (result.empty):
        click.secho("No mean distance found for bead atom, using default value.", fg='yellow')
        l0 = 5.5
    else : 
        l0 = result["Mean"].item()    
    click.secho(f"Mean distance for bead atom {args.bead_atom}: {l0}", fg='green')
    optimizer_classes = {
        "rasp": BeadSpringRASPOptimizer,
        "dfire": BeadSpringDFIREOptimizer,
        "rsRNASP": BeadSpringRsRNASPOptimizer
    }
    
    OptClass = optimizer_classes[args.score]

    opt = OptClass(
            sequence = args.sequence,
            output_path=output_path,
            noise_coords=args.noise_coords,
            bead_atom=args.bead_atom,
            k=args.k,
            l0=l0,
            verbose=args.verbose,
            patience_locale=args.patience_locale,
            patience_globale=args.patience_globale,
            min_delta=args.min_delta,
            taux_refroidissement=args.taux_refroidissement,
            bruit_min=args.bruit_min,
            export_cif=args.cif,
            score_weight=args.score_weight
        )

    # 4. Optimization run
    start_time = time.perf_counter()
    opt.run_optimization()
    end_time = time.perf_counter()
    click.secho(f"Execution time: {end_time - start_time:.2f} seconds", fg='yellow', bold=True)
if __name__ == "__main__":
    main()
