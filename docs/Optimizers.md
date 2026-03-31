# Les Optimiseurs (Dossier `classe/`)

Le projet repose sur deux philosophies d'optimisation distinctes, toutes deux implémentées avec PyTorch pour le calcul tensoriel et l'autodifférenciation.

## 1. Modèle de Perles : `BeadSpringOptimizer.py`

Contrairement à une approche tout-atome, ce modèle réduit chaque nucléotide à une perle unique (généralement l'atome **C3'**). C'est un modèle de physique mésoscopique extrêmement rapide.

### A. Potentiels de Cohésion (Physique)
Pour maintenir l'intégrité de la chaîne d'ARN sans structure atomique complète, on utilise deux forces classiques :
1. **WCA (Weeks-Chandler-Andersen)** : Une version purement répulsive du potentiel de Lennard-Jones. Elle empêche les perles de s'interpénétrer (volume exclu).
2. **FENE (Finitely Extensible Nonlinear Elastic)** : Un ressort non-linéaire qui lie les perles entre elles. Il empêche la chaîne de se briser tout en limitant l'extension maximale des liaisons.

### B. Potentiel Statistique
L'énergie globale est complétée par un score issu de **RASP** ou **DFIRE-RNA**, calculé uniquement sur les distances entre les perles sélectionnées.
- **Classes dérivées** : `BeadSpringRASPOptimizer` et `BeadSpringDFIREOptimizer`.

---

## 2. Modèle de Corps Rigides : `FullAtomOptimizer.py`

Cette approche traite chaque nucléotide comme une entité rigide indéformable. C'est l'approche privilégiée pour un raffinement de haute précision.

### A. Paramétrage Géométrique (6 DOF)
Chaque nucléotide possède 6 degrés de liberté optimisables :
- **Translation (3D)** : Position du centre de référence (`ref_coords`).
- **Rotation (3D)** : Orientation spatiale via des angles d'Euler (`rot_angles`).

La position de chaque atome est reconstruite par : $Coords = Ref\_Coords + R \times Offsets$, où les offsets sont les positions relatives internes fixes des atomes du nucléotide.

### B. Contraintes du Squelette
Comme les nucléotides sont des corps rigides séparés, une pénalité de "backbone" est appliquée pour forcer l'atome **O3'** du nucléotide $i$ à rester proche de l'atome **P** du nucléotide $i+1$ (distance cible ~1.6 Å).

---

## 3. Algorithmes de Résolution

Le projet supporte plusieurs algorithmes de descente :
- **Adam** : Robuste pour les larges oscillations et la phase exploratoire.
- **L-BFGS** : Algorithme de quasi-Newton très efficace pour converger précisément vers le minimum local une fois la structure globalement correcte.

### Basin Hopping (Shake)
Pour échapper aux minima locaux, l'optimisation est découpée en **Cycles**. À chaque nouveau cycle, un bruit contrôlé (secousse) est injecté dans les coordonnées et les angles pour explorer de nouvelles configurations.

---

## 4. Intégration des Scores (RASP & DFIRE)

Les classes finales (ex: `FullAtomDFIREOptimizer`) combinent ces modèles physiques avec les matrices de potentiels. Elles utilisent l'**interpolation linéaire** pour rendre les grilles d'énergie discrètes (bins) différentiables, permettant à PyTorch de calculer les gradients nécessaires au déplacement des nucléotides.
