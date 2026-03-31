# Wiki: RNA Structure Optimization

Bienvenue dans la documentation du projet **RNA Structure Optimization**. Ce projet permet de générer et d'optimiser des structures 3D d'ARN à partir de leur séquence, en utilisant des champs de force statistiques comme **DFIRE-RNA** et **RASP**.

L'application propose deux approches distinctes :
1. **Modèle Bead-Spring** : Représentation simplifiée où chaque nucléotide est réduit à une "perle" (typiquement l'atome C3'), optimisée avec des contraintes physiques (WCA, FENE).
2. **Modèle Full-Atom** : Représentation complète où chaque nucléotide est traité comme un corps rigide (6 degrés de liberté) pour une optimisation précise de tous les atomes lourds.

## Structure du projet

Voici les composants clés et fichiers de code présents dans le dossier :

### 1. Le cœur de l'optimisation (Dossier `classe/`)
La logique d'optimisation utilise PyTorch pour accélérer les calculs géométriques et supporter les algorithmes de descente de gradient (**Adam**) et de quasi-Newton (**L-BFGS**).
- **`BeadSpringOptimizer.py`** : Gère l'optimisation par perles avec forces de rappel et potentiels statistiques.
- **`FullAtomOptimizer.py`** : Gère l'optimisation rigide de tous les atomes lourds.
- **`BeadSpringRASPOptimizer.py`** / **`FullAtomDFIREOptimizer.py`** (etc.) : Implémentations spécifiques combinant un modèle et une fonction de score.

### 2. Interfaces Utilisateur (CLI & Scripts)
- **`main_bead_springs.py`** : Point d'entrée principal pour l'optimisation par modèle de perles.
- **`main_full_atom.py`** : Point d'entrée principal pour l'optimisation atomique complète.
- **`launch_bead_springs.sh`** / **`launch_full_atom.sh`** : Scripts Bash pour faciliter l'exécution avec des paramètres par défaut optimisés.

### 3. Parsing des potentiels
Les potentiels statistiques sont gérés par :
- **`parse_dfire_potentials.py`** : Charge et traite la matrice DFIRE-RNA.
- **`parse_rasp_potentials.py`** : Charge et traite les fichiers d'énergie RASP.

### 4. Fonctions Utilitaires
- **`fonction.py`** : Regroupe les fonctions pour la préparation de la structure initiale (via `tleap` d'AmberTools), le nettoyage des PDB et la lecture des fichiers FASTA.

---
Pour plus de détails, consultez les pages suivantes :
- [Les Optimiseurs (Dossier `classe/`)](Optimizers.md)
- [Interfaces et Exécution (CLI & Bash)](CLI_and_Launch.md)
- [Parsing des Potentiels (DFIRE & RASP)](Potentials_Parsing.md)
- [Fonctions Utilitaires et AmberTools](Utils_Functions.md)
