# Le Parsing des Potentiels Statistique (Énergies)

Les algorithmes d'optimisation de l'ARN utilisent des **champs de forces (fonctions de score paramétriques) statistiques**. Plutôt que d'utiliser des lois physiques, ces scores cherchent à reproduire les repliements vus dans la nature en observant géométriquement des bases de données de cristaux (ex: PDB). Plus nous nous rapprochons de la géométrie de ces bases, plus le "score d'énergie" et l'enthalpie sont théoriquement bas.

Le projet manipule les matrices **DFIRE** et **RASP**.

### 1. `parse_dfire_potentials.py`
Le module DFIRE cherche l'information d'énergies mutuelles entre atomes selon des "types" très précis. Ce script permet la jonction entre le format brute de données (matrice ou PDB) et le programme python.
- `load_dfire_potentials()` : Lit la matrice statistique `matrice_dfire.dat` contenue dans le dossier `potentials/` à l'aide de numpy `loadtxt`.
- `get_dfire_type(atom_name, res_name)` : Les règles exactes DFIRE stipulent qu'un atome se nomme en combinant la base (réduite à A, U, C, G) et son code d'atome (ex: O2', N1). Les atomes d'hydrogènes (`'H*'`) ne sont pas comptés.
- `calculate_dfire_score()` : Outil indépendant et purement CPU testant toutes les paires interatomiques avec $O(N^2)$ complexités dans la structure PDB. Calcule l'écart mathématique $r$, tronque ce potentiel a un maximum empirique rigide ($\ge 19.6$ Å) et ajoute un score d'énergies si ce potentiel est franchi. *Note: L'optimiseur PyTorch n'utilise pas cette méthode séquentielle pour s'entraîner (elle serait bien trop lente sans parallélisation GPU et Backward computation graph de PyTorch).*

### 2. `parse_rasp_potentials.py`
Le fonctionnement de RASP, plus complexe et puissant que DFIRE, est implémenté nativement de manière très structurée. Il compte sur une liste finie stricte de **23 Types** d'atomes (du backbone comme O5', ou des bases comme N1/Pyr ou C6(A)).
- Le gigantesque dictionnaire statique nommé `RASP_ATOM_TYPES` mappe manuellement au dixième d'index près l'ensemble des cas possibles des codes atomiques RASP sur leur tuple de coordonnées `(k, t1, t2, dist)`.
- `load_rasp_potentials()` : Parse logiquement une matrice multidimensionnelle (Tensor de 4 Dimensions $[k, Type_A, Type_B, Distance]$) où *k* est la distance en séquence (sép. sur la structure primaire) de 2 résidus.
- `get_rasp_type()` : Déclinée avec un paramètre (par défaut Full-Atoms), cette fonction est capable d'être appelée de façon "dégradée" via l'argument `type_RASP="c3"` afin de ne renvoyer une valeur numérique qu'au seul élément $C3'$, ce qui divise la taille des tenseurs sur machine par cent (utile pour `RNA_RASP_C3_Optimizer`).
- `calculer_score_rasp_smooth()`: Variante plus souple pour obtenir de l'énergie de tests de validations hors de l'optimiseur. Elle exploite l'**Interpolation Cubique (Mathématiques/Splines)**. Cette opération permet d’évaluer la courbure temporelle et l'intervalle spatial au sein des seuils fixés "discrets" RASP (qui procède par tranche de 1 Angström). Cela comble d'importantes erreurs d’évaluation énergétique si nos atomes atterrissent *pile au milieu* d'un bucket RASP (`dist = 1.6`) ce qui rend la courbe d'énergie totalement différentiable mathématiquement.
