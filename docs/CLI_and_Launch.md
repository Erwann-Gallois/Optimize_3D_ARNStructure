# Interfaces et Lancement de l'Optimisation

Le projet propose différentes manières d'exécuter l'optimisation selon votre besoin (interactif, batch, scriping).

### 1. Le script Bash: `launch_opt.sh`
C'est le moyen le plus simple et ergonomique pour démarrer l'application si l'on ne souhaite pas entrer de longs arguments dans un terminal.
- Ouvre un menu interactif texte dans votre terminal.
- Permet de choisir via touches de clavier "1" ou "2" de donner sa structure en renseignant la séquence manuellement dans le terminal ou via le chemin du fichier FASTA.
- Demande quel moteur d'énergie prioriser (RASP ou DFIRE).
- Propose de changer les `cycles` (tours du Simulated Annealing) ou d'`epochs` (pas de descente de gradients).
- Configure un nom de fichier output.
- Concatène toutes ces réponses pour formuler automatiquement et lancer la commande d'exécution à `cli.py`.

### 2. Le point d'entrée universel : `cli.py`
Il s'agit de l'exécutable formel de l'application (Command Line Interface). Propulsé par la librairie logicielle native `argparse`.
Il gère : 
1. La vérification syntaxique des arguments donnés (`--score`, `--cycles`, `-s`, `-f`).
2. S'il reçoit un fichier `.fasta`, appelle la librairie `Biopython` / `SeqIO` pour en extraire la trame textuelle de manière robuste.
3. Il crée toujours une structure canonique temporaire `fichier_arn/initial_{timestamp}.pdb` en appelant la fonction Amber `generer_first_structure()`.
4. Il instancie ensuite la bonne classe d'optimisation PyTorch (`RNA_RASP_Optimizer` ou `RNA_DFIRE_Optimizer`, notez que dans ce fichier CLI, seuls les optimizers *Full-Atoms* sont appelés par défaut).
5. Mesure le temps et gère l'enregistrement dans le dossier `resultat/`.

