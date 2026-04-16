import click
import os
import sys
from main_bead_springs import main as run_bead_springs
from main_full_atom import main as run_full_atom
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def display_summary(args, output_path):
    console = Console()
    
    # Create the table
    table = Table(title="Optimization Parameters Summary", title_style="bold magenta")

    # Define columns
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    if args.method_name == "bead-springs":
        table.add_row("Sequence", str(args.input_val))
        table.add_row("Scoring Function", str(args.score).upper())
        table.add_row("Output path", str(output_path))
        table.add_row("Patience (Local/Global)", f"{args.patience_locale} / {args.patience_globale}")
        table.add_row("Min Delta", str(args.min_delta))
        table.add_row("Cooling Rate", str(args.taux_refroidissement))
        table.add_row("Minimum Noise", f"{args.bruit_min} Å")
        table.add_row("Initial Noise", f"{args.noise_coords} Å")
        table.add_row("Stiffness Constant K", str(args.k))
        table.add_row("Score Weight", str(args.score_weight))
    else:
        table.add_row("Sequence", str(args.input_val))
        table.add_row("Scoring Function", str(args.score).upper())
        table.add_row("Output", str(output_path))
        table.add_row("Patience (Local/Global)", f"{args.patience_locale} / {args.patience_globale}")
        table.add_row("Min Delta", str(args.min_delta))
        table.add_row("Cooling Rate", str(args.taux_refroidissement))
        table.add_row("Minimum Noise", f"{args.bruit_min} Å")
        table.add_row("Initial Noise", f"{args.noise_coords} Å")
        table.add_row("Angle Noise", f"{args.noise_angles} rad")
        table.add_row("Backbone Weight", str(args.backbone_weight))
    # Display 
    console.print(Panel(table, expand=False))

def set_argv(args_list):
    """Utility to simulate CLI arguments for argparse"""
    sys.argv = [sys.argv[0]] + args_list

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # This message will ALWAYS display first
    click.secho("\n" + "="*50, fg='cyan')
    click.secho("   3D RNA Optimization Launch Interface", bold=True, fg='cyan')
    click.secho("="*50 + "\n", fg='cyan')
    
    # If python main.py is typed alone, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command(help="Launch optimizations with the entered parameters. You can specify arguments to avoid interactive mode.")
@click.option('--method', 
              type=click.Choice(['1', '2']), 
              prompt="Choose an optimization method:\n1) Bead-springs\n2) Full atoms\n Choice",
              default='1', help="Optimization method: 1 (Bead-springs) or 2 (Full atoms).")
@click.option('--input-type',default = '1', type=click.Choice(["1", "2"]), prompt="Input type:\n1) Sequence\n2) PDB file\nChoice", help="Input format: 1 (RNA Sequence) or 2 (Path to a PDB file).")
@click.option('--input-val', prompt="Value (Sequence or file path)", help="The sequence itself (e.g., ACGU) or the path to the PDB file.")
@click.option('--score', type=click.Choice(["1", "2", "3", "4"]), default='4', prompt="Scoring function:\n1) RASP\n2) DFIRE\n3) rsRNASP\n4) All\nChoice", help="Scoring function used: 1 (RASP), 2 (DFIRE), 3 (rsRNASP), 4 (Launch with all).")
@click.option('--output', default=None, help="Output file path (optional, automatically calculated if not provided).")
@click.option('--cif', is_flag=True, prompt="Export in CIF format?", default=False, help="Export the final optimized structure in CIF format in addition to PDB.")
@click.option('--verbose', is_flag=True, prompt="Activate verbose mode?", default=False, help="Activate detailed logs and progress bars during optimization.")
@click.option('--confirm', is_flag=True, prompt="Show summary and confirm?", default=True, help="Shows parameter summary and requires manual confirmation before launch.")

# Shared parameters with intelligent defaults
@click.option('--patience-locale', default=100, prompt=True, type=int, help="Local patience: number of local iterations without improvement before applying a shake.")
@click.option('--patience-globale', default=5, prompt=True, type=int, help="Global patience: number of consecutive shakes (macro-cycles) without a new record before termination.")
@click.option('--min-delta', default=1e-4, prompt=True, type=float, help="Min delta: threshold below which energy variation is considered negligible.")
@click.option('--taux-refroidissement', default=0.85, prompt=True, type=float, help="Cooling rate: noise reduction ratio at each cycle (-15% = 0.85).")
@click.option('--bruit-min', default=0.01, prompt=True, type=float, help="Minimum noise: below this shake threshold, global optimization ends.")
@click.option('--noise-coords', default=1.5, prompt=True, type=float, help="Initial noise (spatial amplitude of coordinate perturbation).")

# Specific parameters (prompts managed manually in launch)
@click.option('--k', default=10.0, help="FENE spring stiffness constant for bonds (Bead-springs model).", type=float)
@click.option('--backbone-weight', default=100, help="Backbone steric constraint weight P-O5'-C5' (Full-atom model).", type=int)
@click.option('--bead-atom', default="C3'", help="Representative bead atom (e.g., C3', P) for coarse graining (Bead-springs model).")
@click.option('--score_weight', default=1.0, help="Global scaling factor applied to the scoring function.", type=float, show_default=True)
@click.option("--noise-angles", default=0.2, type=float, help="Initial noise (angular perturbation amplitude in radians, e.g., 0.2 ~ 15°).")

def launch(method, input_type, input_val, score, cif, confirm, **kwargs):
    """Launch optimizations with the entered parameters."""
    method_mapping = {"1": "bead-springs", "2": "full_atom"}
    method_name = method_mapping[method]

    # --- DYNAMIC PROMPT MANAGEMENT ---
    if method_name == "bead-springs":
        kwargs['k'] = click.prompt("Stiffness constant k (spring)", default=kwargs.get('k', 10.0), type=float)
        kwargs['bead_atom'] = click.prompt("Representative atom (bead-atom)", default=kwargs.get('bead_atom', "C3'"))
        kwargs['score_weight'] = click.prompt("Score weight", default=kwargs.get('score_weight', 1.0), type=float)
    else:
        kwargs['backbone_weight'] = click.prompt("Backbone weight", default=kwargs.get('backbone_weight', 100), type=int)
        kwargs['noise_angles'] = click.prompt("Angle noise", default=kwargs.get('noise_angles', 0.2), type=float)

    input_type_mapping = {"1": "seq", "2": "file"}
    input_type = input_type_mapping[input_type]

    # Prepare scores to process
    score_mapping = {"1": "rasp", "2": "dfire", "3": "rsRNASP", "4": "all"}
    score = score_mapping[score]
    scores_to_run = ["rasp", "dfire", "rsRNASP"] if score == "all" else [score]
    
    # Construct input flag (-s or -f)
    input_flag = "-s" if input_type == "seq" else "-f"
    args = [input_flag, input_val]


    # --- UNIQUE VALIDATION ---
    if confirm and score == "all":
        # Create a dummy object for the summary
        class Config: pass
        cfg = Config()
        cfg.method_name= method_name
        cfg.input_val, cfg.score = input_val, score
        cfg.patience_locale, cfg.patience_globale = kwargs['patience_locale'], kwargs['patience_globale']
        cfg.noise_coords, cfg.bruit_min = kwargs['noise_coords'], kwargs['bruit_min']
        cfg.taux_refroidissement, cfg.min_delta = kwargs['taux_refroidissement'], kwargs['min_delta']
        cfg.score_weight, cfg.k, cfg.bead_atom = kwargs.get('score_weight', 0.0), kwargs.get('k', 0.0), kwargs.get('bead_atom', "N/A")
        cfg.backbone_weight, cfg.noise_angles = kwargs.get('backbone_weight', 0), kwargs.get('noise_angles', 0.0)
        
        display_summary(cfg, kwargs.get('output') or "Auto-generated")
        
        if not click.confirm(click.style("\nLaunch optimization?", fg='magenta', bold=True), default=True):
            click.secho("Cancelled by user.", fg='red')
            return
    else:
        if confirm: args.append("--confirm")
    
    for current_score in scores_to_run:
        args = args + ["--score", current_score]
        click.secho("\n" + "="*50, fg='cyan')
        click.secho(f"\nStarting: {method_name.upper()} | SCORE: {current_score}", fg='cyan', bold=True)
        click.secho("\n" + "="*50, fg='cyan')
        
        if cif: args.append("--cif")
        if kwargs['verbose']: args.append("-v")
        
        # Ajout des paramètres numériques
        args += ["--patience-locale", str(kwargs['patience_locale'])]
        args += ["--patience-globale", str(kwargs['patience_globale'])]
        args += ["--min-delta", str(kwargs['min_delta'])]
        args += ["--taux-refroidissement", str(kwargs['taux_refroidissement'])]
        args += ["--bruit-min", str(kwargs['bruit_min'])]
        args += ["--noise-coords", str(kwargs['noise_coords'])]
        if kwargs.get('output'):
            args += ["--output", str(kwargs['output'])]
        if method_name == "bead-springs":
            args += ["--k", str(kwargs['k'])]
            args += ["--bead-atom", kwargs['bead_atom']]
            args += ["--score_weight", str(kwargs['score_weight'])]
            set_argv(args)
            run_bead_springs()
        else:
            args += ["--backbone-weight", str(kwargs['backbone_weight'])]
            args += ["--noise-angles", str(kwargs['noise_angles'])]
            set_argv(args)
            run_full_atom()

if __name__ == '__main__':
    cli()