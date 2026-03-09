# Pseudo-code : Classe RNA_RASP_Rigid

Cette classe implémente une optimisation de structure ARN en traitant chaque nucléotide comme un corps rigide (Rigid Body). L'objectif est de minimiser l'énergie potentielle RASP tout en maintenant l'intégrité de la chaîne (pénalité backbone).

## 1. Initialisation (`__init__`)
- Charger les paramètres : PDB, taux d'apprentissage (lr), type de potentiel, nombre de cycles, etc.
- **`load_dict_potentials()`** : Charge les fichiers d'énergie RASP depuis un dictionnaire vers un tenseur 4D `(k, type1, type2, distance)`.
- **`convert_pdb_to_rigid_tensors()`** : Prépare les données pour l'optimisation.

---

## 2. Préparation des données rigides
- Filtrer les atomes reconnus par RASP.
- Pour chaque nucléotide (résidu) :
    - Définir une **coordonnée de référence** (ex: atome C3').
    - Calculer les **offsets** : vecteur relatif de chaque atome par rapport à sa référence.
- Identifier les paires d'atomes pour le calcul de l'énergie (basé sur la séparation séquentielle `k`).
- Identifier les liaisons backbone (ex: distance O3' -> P) pour la contrainte physique.

---

## 3. Variables Optimisables (Paramètres PyTorch)
1. **`ref_coords`** : Tenseur des positions (x, y, z) de chaque centre de nucléotide (Translation).
2. **`rot_angles`** : Tenseur des 3 angles d'Euler pour chaque nucléotide (Rotation).

---

## 4. Reconstruction de la structure (`get_current_full_coords`)
Pour chaque nucléotide :
1. Générer une **matrice de rotation 3x3** à partir des angles d'Euler.
2. Appliquer cette rotation aux **offsets** pré-calculés.
3. Additionner la position de référence (**translation**).
*Résultat : Coordonnées cartésiennes complètes tout en préservant la géométrie interne de chaque nucléotide.*

---

## 5. Calcul du Score (`calculate_detailed_scores`)
### A. Score RASP
- Pour chaque paire d'atomes :
    - Calculer la distance euclidienne.
    - Interpoler l'énergie dans le tenseur de potentiel (entre les bins de distance `d` et `d+1`).
    - Somme des énergies pour toutes les paires valides (< 19Å).

### B. Pénalité Backbone
- Calculer la distance entre les atomes terminaux connectés (O3' du résidu $i$ et P du résidu $i+1$).
- Appliquer une pénalité quadratique si la distance s'écarte de la cible (1.61Å).

---

## 6. Boucle d'Optimisation (`run_optimization`)
Pour chaque **Cycle** (Sauts globaux) :
    Pour chaque **Epoch** (Descente de gradient) :
    1. Calculer les coordonnées actuelles.
    2. Calculer le Loss (`RASP + Backbone`).
    3. Backpropagation des gradients vers `ref_coords` et `rot_angles`.
    4. Mise à jour via l'optimiseur **Adam**.
    5. Sauvegarder la meilleure conformation si le score est amélioré.

    **Action "Shake" (Fin de cycle) :**
    - Repartir du meilleur état trouvé.
    - Ajouter un **bruit gaussien** aux positions et aux angles pour sortir des minima locaux.
    - Réinitialiser l'optimiseur (effacer l'inertie d'Adam).

---

## 7. Finalisation
- Restaurer la meilleure conformation globale.
- Sauvegarder le résultat final dans un nouveau fichier PDB.
