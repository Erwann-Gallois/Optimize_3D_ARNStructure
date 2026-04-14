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
    
    # Création du tableau
    table = Table(title="Résumé des Paramètres d'Optimisation", title_style="bold magenta")

    # Définition des colonnes
    table.add_column("Paramètre", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="green")

    if args.method_name == "bead-springs":
        table.add_row("Séquence", str(args.input_val))
        table.add_row("Fonction de Score", str(args.score).upper())
        table.add_row("Sortie", str(output_path))
        table.add_row("Patience (Local/Globale)", f"{args.patience_locale} / {args.patience_globale}")
        table.add_row("Min Delta", str(args.min_delta))
        table.add_row("Taux de Refroidissement", str(args.taux_refroidissement))
        table.add_row("Bruit minimum", f"{args.bruit_min} Å")
        table.add_row("Bruit initial", f"{args.noise_coords} Å")
        table.add_row("Constante K", str(args.k))
        table.add_row("Poids du Score", str(args.score_weight))
    else:
        table.add_row("Séquence", str(args.input_val))
        table.add_row("Fonction de Score", str(args.score).upper())
        table.add_row("Sortie", str(output_path))
        table.add_row("Patience (Local/Globale)", f"{args.patience_locale} / {args.patience_globale}")
        table.add_row("Min Delta", str(args.min_delta))
        table.add_row("Taux de Refroidissement", str(args.taux_refroidissement))
        table.add_row("Bruit minimum", f"{args.bruit_min} Å")
        table.add_row("Bruit initial", f"{args.noise_coords} Å")
        table.add_row("Bruit sur les angles", f"{args.noise_angles} rad")
        table.add_row("Backbone Weight", str(args.backbone_weight))
    # Affichage 
    console.print(Panel(table, expand=False))

def set_argv(args_list):
    """Utilitaire pour simuler les arguments CLI pour argparse"""
    sys.argv = [sys.argv[0]] + args_list

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # Ce message s'affichera TOUJOURS en premier
    click.secho("\n" + "="*50, fg='cyan')
    click.secho("   3D RNA Optimization Launch Interface", bold=True, fg='cyan')
    click.secho("="*50 + "\n", fg='cyan')
    
    # Si on tape juste 'python main.py', on affiche l'aide
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command(help="Lance les optimisations avec les paramètres saisis. Vous pouvez préciser les arguments pour éviter le mode interactif.")
@click.option('--method', 
              type=click.Choice(['1', '2']), 
              prompt="Choose a optimization method :\n1) Bead-springs\n2) Full atoms\n Choice",
              default='1', help="Méthode d'optimisation : 1 (Bead-springs) ou 2 (Full atoms).")
@click.option('--input-type',default = '1', type=click.Choice(["1", "2"]), prompt="Type d'entrée :\n1) Séquence\n2) Fichier PDB\nChoice", help="Format de l'entrée : 1 (Séquence ARN) ou 2 (Chemin vers un Fichier PDB).")
@click.option('--input-val', prompt="Valeur (Séquence ou Chemin du fichier)", help="La séquence elle-même (ex: ACGU) ou le chemin du fichier PDB.")
@click.option('--score', type=click.Choice(["1", "2", "3", "4"]), default='4', prompt="Fonction de score :\n1) RASP\n2) DFIRE\n3) rsRNASP\n4) All\nChoice", help="Fonction de score utilisée : 1 (RASP), 2 (DFIRE), 3 (rsRNASP), 4 (Lancer avec toutes).")
@click.option('--output', default=None, help="Chemin du fichier de sortie (optionnel, calculé automatiquement si non renseigné).")
@click.option('--cif', is_flag=True, prompt="Exporter en format CIF ?", default=False, help="Export de la structure finale optimisée au format CIF en plus du PDB.")
@click.option('--verbose', is_flag=True, prompt="Activate verbose mode ?", default=False, help="Active un log détaillé et les jauges de progression lors de l'optimisation.")
@click.option('--confirm', is_flag=True, prompt="Afficher le résumé et confirmer ?", default=True, help="Affiche le résumé des paramètres et requiert une confirmation manuelle avant le lancement.")

# Paramètres partagés avec valeurs par défaut intelligentes
@click.option('--patience-locale', default=100, prompt=True, type=int, help="Patience locale : nombre d'itérations locales sans amélioration avant d'appliquer un shake.")
@click.option('--patience-globale', default=5, prompt=True, type=int, help="Patience globale : nombre de shakes (macro-cycles) consécutifs sans nouveau record avant l'arrêt.")
@click.option('--min-delta', default=1e-4, prompt=True, type=float, help="Min delta : seuil en dessous duquel la variation d'énergie est jugée négligeable.")
@click.option('--taux-refroidissement', default=0.85, prompt=True, type=float, help="Taux de refroidissement : ratio de réduction du bruit d'exploration à chaque cycle (-15% = 0.85).")
@click.option('--bruit-min', default=0.01, prompt=True, type=float, help="Bruit minimum limitant, sous ce seuil de shake, l'optimisation globale se termine.")
@click.option('--noise-coords', default=1.5, prompt=True, type=float, help="Bruit initial (amplitude spatiale de perturbation aux coordonnées).")

# Paramètres spécifiques (prompts gérés manuellement dans launch)
@click.option('--k', default=10.0, help="Constante de raideur de ressort FENE pour les liaisons (Modèle Bead-springs).", type=float)
@click.option('--backbone-weight', default=100, help="Poids d'ancrage des contraintes stériques du squelette P-O5'-C5' (Modèle Full-atom).", type=int)
@click.option('--bead-atom', default="C3'", help="Atome représentatif du grain (ex: C3', P) pour le coarse graining (Modèle Bead-springs).")
@click.option('--score_weight', default=1.0, help="Facteur d'échelle global appliqué à la fonction de score.", type=float, show_default=True)
@click.option("--noise-angles", default=0.2, type=float, help="Bruit initial (amplitude de perturbation sur les angles en radians, par ex 0.2 ~ 15°).")

def launch(method, input_type, input_val, score, cif, confirm, **kwargs):
    """Lance les optimisations avec les paramètres saisis."""
    method_mapping = {"1": "bead-springs", "2": "full_atom"}
    method_name = method_mapping[method]

    # --- GESTION DYNAMIQUE DES PROMPTS ---
    if method_name == "bead-springs":
        kwargs['k'] = click.prompt("Constante k (ressort)", default=kwargs.get('k', 10.0), type=float)
        kwargs['bead_atom'] = click.prompt("Atome représentatif (bead-atom)", default=kwargs.get('bead_atom', "C3'"))
        kwargs['score_weight'] = click.prompt("Poids du score", default=kwargs.get('score_weight', 1.0), type=float)
    else:
        kwargs['backbone_weight'] = click.prompt("Poids du backbone", default=kwargs.get('backbone_weight', 100), type=int)
        kwargs['noise_angles'] = click.prompt("Bruit sur les angles", default=kwargs.get('noise_angles', 0.2), type=float)

    input_type_mapping = {"1": "seq", "2": "file"}
    input_type = input_type_mapping[input_type]

    # Préparation des scores à traiter
    score_mapping = {"1": "rasp", "2": "dfire", "3": "rsRNASP", "4": "all"}
    score = score_mapping[score]
    scores_to_run = ["rasp", "dfire", "rsRNASP"] if score == "all" else [score]
    
    # Construction de l'argument d'entrée (-s ou -f)
    input_flag = "-s" if input_type == "seq" else "-f"

    # --- VALIDATION UNIQUE ---
    if confirm and score == "all":
        # Création d'un objet factice pour le résumé
        class Config: pass
        cfg = Config()
        cfg.method_name= method_name
        cfg.input_val, cfg.score = input_val, score
        cfg.patience_locale, cfg.patience_globale = kwargs['patience_locale'], kwargs['patience_globale']
        cfg.noise_coords, cfg.bruit_min = kwargs['noise_coords'], kwargs['bruit_min']
        cfg.taux_refroidissement, cfg.min_delta = kwargs['taux_refroidissement'], kwargs['min_delta']
        cfg.score_weight, cfg.k, cfg.bead_atom = kwargs.get('score_weight', 0.0), kwargs.get('k', 0.0), kwargs.get('bead_atom', "N/A")
        cfg.backbone_weight, cfg.noise_angles = kwargs.get('backbone_weight', 0), kwargs.get('noise_angles', 0.0)
        
        display_summary(cfg, kwargs.get('output') or "Auto-généré")
        
        if not click.confirm(click.style("\n🚀 Lancer l'optimisation ?", fg='magenta', bold=True), default=True):
            click.secho("❌ Annulé par l'utilisateur.", fg='red')
            return
    else:
        if kwargs['confirm']: args.append("--confirm")
    
    for current_score in scores_to_run:
        click.secho("\n" + "="*50, fg='cyan')
        click.secho(f"\nDémarrage : {method_name.upper()} | SCORE : {current_score}", fg='cyan', bold=True)
        click.secho("\n" + "="*50, fg='cyan')
        
        # Construction de la liste d'arguments pour argparse
        args = [input_flag, input_val, "--score", current_score]
        
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