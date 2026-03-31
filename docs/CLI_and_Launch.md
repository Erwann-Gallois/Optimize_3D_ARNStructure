# Interfaces et Lancement de l'Optimisation

Le projet propose deux points d'entrée principaux selon le modèle d'optimisation souhaité.

## 1. Points d'entrée (Python)

### `main_bead_springs.py`
Optimisation via le modèle **Bead-Spring** (chaîne de perles).
```bash
python main_bead_springs.py -s ACGU --score dfire --epochs 50 --cycles 20
```

### `main_full_atom.py`
Optimisation via le modèle **Full-Atom** (corps rigides).
```bash
python main_full_atom.py -f sequence.fasta --score rasp --lr 0.2
```

## 2. Arguments de la ligne de commande (CLI)

Les deux scripts partagent des arguments communs, mais possèdent aussi des options spécifiques.

### Arguments communs :
- `-s`, `--sequence` : Séquence ARN directe (ex: `ACGU`).
- `-f`, `--fasta` : Chemin vers un fichier FASTA.
- `--score {rasp, dfire}` : Fonction de score statistique à utiliser (défaut: `dfire`).
- `-o`, `--output` : Chemin du fichier PDB de sortie.
- `--epochs` : Nombre d'époques par cycle d'optimisation (défaut: 50).
- `--cycles` : Nombre de cycles de "basin hopping" (défaut: 20).
- `--lr` : Taux d'apprentissage (Learning Rate, défaut: 0.2).
- `--noise-coords` : Intensité du bruit ajouté aux coordonnées à chaque cycle (défaut: 0.5).
- `-v`, `--verbose` : Active l'affichage détaillé pendant l'optimisation.

### Spécifiques à Bead-Spring :
- `--k` : Constante de raideur des ressorts (défaut: 40.0).
- `--l0` : Distance d'équilibre entre les perles (défaut: 5.5).
- `--bead-atom` : Atome utilisé comme centre de la perle (défaut: `C3'`).

### Spécifiques à Full-Atom :
- `--backbone-weight` : Poids de la pénalité sur la connectivité du squelette (défaut: 100).
- `--noise-angles` : Intensité du bruit ajouté aux angles de rotation à chaque cycle (défaut: 0.2).

## 3. Scripts de lancement (Bash)

Pour faciliter l'exécution sans mémoriser tous les arguments, deux scripts assistants sont fournis :
- **`launch_bead_springs.sh`** : Prépare une exécution optimisée pour le modèle de perles.
- **`launch_full_atom.sh`** : Prépare une exécution optimisée pour le modèle atomique complet.

Ces scripts gèrent automatiquement la création des dossiers `fichier_arn/` et `resultat/` si nécessaire.

