# Wiki: Optimize_3D_ARNStructure

Bienvenue dans la documentation du projet **Optimize_3D_ARNStructure**. Ce projet a pour but de générer et d'optimiser des structures 3D d'ARN à partir de leur séquence, en utilisant des champs de force statistiques (potentiels empiriques) comme **DFIRE-RNA** et **RASP**.

L'application permet de :
1. Générer une structure 3D initiale (ARN droit et rigide) via **AmberTools** (`tleap`).
2. Nettoyer et standardiser la structure PDB générée.
3. Optimiser la géométrie (translation et rotation des nucléotides traités comme des corps rigides) grâce au framework d'optimisation PyTorch (`RNA_Optimizer.py` et ses dérivés).
4. Évaluer l'énergie de la conformation générée via les potentiels DFIRE et RASP.

## Structure du projet

Voici les composants clés et fichiers de code présents dans le dossier :

### 1. Le cœur de l'optimisation (Dossier `classe/`)
La logique d'optimisation est conçue en programmation orientée objet, basée sur PyTorch pour accélérer le calcul géométrique et la descente de gradient avec `Adam`.
- **`RNA_Optimizer.py`** : Classe mère qui gère la préparation des tenseurs géométriques, la rotation/translation (matrices d'Euler), le calcul des pénalités (clash VDW, contraintes de squelette `O3'-P`) et la boucle d'optimisation `Adam`.
- **`RNA_DFIRE_Optimizer.py`** et **`RNA_RASP_Optimizer.py`** : Optimiseurs fonctionnant en "Full-Atom" (tous les atomes lourds) en interpolant les énergies DFIRE ou RASP.
- **`RNA_DFIRE_C3_Optimizer.py`** et **`RNA_RASP_C3_Optimizer.py`** : Variantes optimisées pour la performance. Elles calculent les scores énergétiques *uniquement* sur les atomes C3', tout en maintenant l'intégrité du backbone avec les atomes O3'/P, puis reconstruisent tous les atomes de la molécule à la fin.
- **`RNA_RASP_Rigid_parallel.py`** : Une variante de l'optimiseur RASP préparée pour tourner de manière optimisée sur processeur (multiprocessing CPU) plutôt que sur GPU.

### 2. Interfaces Utilisateur (CLI & Scripts)
- **`cli.py`** : Interface en ligne de commande principale permettant de choisir la fonction de score, d'indiquer une séquence (ou un fichier FASTA), et de régler les hyperparamètres (learning rate, epochs, cycles).
- **`launch.py`** : Petit script interactif basique en python pour lancer une exécution rapide (demande la séquence et le score dans la console).
- **`launch_opt.sh`** : Script Bash interactif complet qui sert de "wizard" (assistant) pour lancer `cli.py` en posant des questions pas-à-pas à l'utilisateur.

### 3. Parsing des potentiels
Les potentiels statistiques (matrices d'énergie) sont lus et traités par ces scripts :
- **`parse_dfire_potentials.py`** : Charge la matrice DFIRE (`matrice_dfire.dat`), assigne à chaque atome PDB son type DFIRE correspondant, et contient des fonctions pour calculer l'énergie globale DFIRE.
- **`parse_rasp_potentials.py`** : Charge les fichiers d'énergie RASP (`.nrg`), mappe les atomes selon les 23 types spécifiques de l'article RASP, et calcule l'énergie en tenant compte de la séparation séquentielle et des limites de distance.

### 4. Fonctions Utilitaires
- **`fonction.py`** : Regroupe toutes les fonctions utilitaires pour préparer la structure initiale :
  - `generer_arn_droit()` : Appelle de façon programmatique `tleap` (AmberTools) pour obtenir un PDB canonique à partir de la séquence.
  - `enlever_hydrogene()` : Supprime les atomes d'hydrogène du PDB (optimisation full-atom "lourd" uniquement).
  - `fix_amber_pdb()` : Répare des atomes manquants en bout de chaîne (ex: reconstruction d'OP3 manquant dans AmberTools).
  - `view_structure()` : Outil de visualisation NGLView (pour les notebooks Jupyter).
  
---
Pour entrer dans les détails, naviguez vers les pages wiki suivantes :
- [Les Optimiseurs (Dossier `classe/`)](Optimizers.md)
- [Interfaces et Exécution (CLI & Bash)](CLI_and_Launch.md)
- [Parsing des Potentiels (DFIRE & RASP)](Potentials_Parsing.md)
- [Fonctions Utilitaires et AmberTools](Utils_Functions.md)
