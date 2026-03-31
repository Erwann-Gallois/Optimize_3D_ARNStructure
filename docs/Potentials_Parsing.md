# Le Parsing des Potentiels Statistiques (Énergies)

Les algorithmes d'optimisation utilisent des **champs de forces statistiques** (potentiels empiriques). Ces scores cherchent à reproduire les repliements vus dans la nature en observant la géométrie des structures connues dans la PDB.

Le projet intègre les matrices de potentiels **DFIRE-RNA** et **RASP**.

### 1. `parse_dfire_potentials.py`
Le module DFIRE calcule les énergies d'interaction basées sur une distance normalisée par un état de référence (gaz parfait de molécules).
- **`load_dfire_potentials()`** : Charge la matrice `matrice_dfire.dat` depuis le dossier `potentials/`.
- **Types d'atomes** : DFIRE définit des types précis combinant le nom du résidu (A, U, C, G) et le nom de l'atome (ex: `G_N1`, `A_C8`).
- **Interpolation** : Pour rendre le potentiel différentiable (essentiel pour la descente de gradient), le code utilise une **interpolation linéaire** entre les "bins" de distance (0.5 Å). Cela permet à PyTorch de calculer un gradient continu même entre deux valeurs discrètes de la matrice.

### 2. `parse_rasp_potentials.py`
Le potentiel RASP est plus complexe et prend en compte la séparation séquentielle entre les résidus.
- **`load_rasp_potentials()`** : Charge les fichiers d'énergie RASP et construit un tenseur à 4 dimensions : `[séparation_k, type_A, type_B, distance]`.
- **Séparation Séquentielle** : L'énergie varie selon que les nucléotides sont proches ou éloignés dans la séquence (6 catégories de 0 à >4).
- **Types RASP** : Utilise 23 types d'atomes spécifiés par le modèle original.
- **Modularité** : Le parser supporte à la fois le mode "Full-Atom" (tous les types) et le mode "C3'" (uniquement l'atome de perle), permettant son utilisation dans les deux modèles d'optimisation.

### 3. Intégration dans les Optimiseurs
Dans les classes d'optimisation (dossier `classe/`), ces potentiels sont convertis en **Tenseurs PyTorch**.
- Les calculs de distance sont vectorisés pour être exécutés massivement sur GPU.
- Des techniques de **"Capping"** (troncature) sont appliquées pour ignorer les interactions au-delà d'une certaine distance (typiquement 20 Å), optimisant ainsi les performances.
