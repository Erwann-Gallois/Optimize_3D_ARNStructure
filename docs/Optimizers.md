# Les Optimiseurs PyTorch (Dossier `classe/`)

Ce dossier rassemble toutes les classes métier qui réalisent l'optimisation géométrique des nucléotides dans l'espace 3D. Le cœur du système repose sur l'approche de la **mécanique du corps rigide (Rigid Body Mechanics)** combinée avec la puissance de calcul tensoriel et d'autograd de **PyTorch**.

## 1. Mécanique du Corps Rigide : `RNA_Optimizer.py`

La classe abstraite `RNA_Optimizer` définit la physique et la géométrie du système. Au lieu de laisser les calculs déplacer librement chaque atome — ce qui détruirait les liaisons chimiques intra-nucléotidiques — l'algorithme regroupe les atomes par nucléotide.

### A. Représentation Géométrique (Tenseurs paramétriques)
Chaque nucléotide est modélisé par des paramètres optimisables au sens du Machine Learning :
- **Un centre de référence (`self.ref_coords`)** : Tenseur paramétrique (`torch.nn.Parameter`) de dimension `(N, 3)` représentant le point de translation de chaque nucléotide (par défaut défini sur l'atome C3').
- **Une orientation spatiale (`self.rot_angles`)** : Paramètre PyTorch de dimension `(N, 3)` représentant les angles d'Euler ($\alpha, \beta, \gamma$) autour des axes X, Y, Z.
- **Des offsets constants (`self.offsets`)** : Les coordonnées locales de chaque atome par rapport à son propre centre de référence. Ces vecteurs spatiaux sont construits par soustraction statique originelle et ne changent *jamais* pendant l'optimisation, garantissant l'intégrité absolue de la base azotée, du sucre et des liaisons covalentes locales.

La position absolue de tout atome est recalculée de manière différentiable à la volée via la fonction `get_current_full_coords()` en appliquant la formule vectorielle : 
$$Coords = Ref\_Coords + R \times Offsets$$ 
où $R$ est la matrice de rotation finale 3x3 propre à chaque nucléotide (issue de la multiplication des matrices d'Euler $Rz \times Ry \times Rx$).

### B. Maintien de la Cohésion Globale (Pénalités de contrainte)
L'optimiseur construit une fonction de perte globale (`loss`) dans `calculate_base_penalties` à minimiser :
1. **La pénalité stérique (Van der Waals Clash)** : On utilise une liste `VDW_RADII` des rayons de Van der Waals (C: 1.7A, P: 1.8A etc.). La distance inter-atomique est contrainte : `diff = torch.clamp(min_dist_vdw - dists, min=0.0)`. Tout rapprochement inférieur au seuil engendre une pénalité sévère et quadratique (`1000.0 * diff**2`) pour repousser instantanément les atomes en superposition spatiale anormale.
2. **La pénalité du Squelette (Backbone O3' - P)** : Pour se comporter comme une véritable chaîne de polymère, on calcule la distance euclidienne entre l'oxygène O3' du nucléotide $i$ et le phosphore P du nucléotide $i+1$. L'algorithme exerce une attraction mathématique (`backbone_weight`) forçant cette distance à tendre vers la cible idéale de **1.61 Å** (longueur de liaison phosphodiester).

### C. L'Algorithme d'Optimisation et le "Shake" (Recuit Simulé)
La méthode `run_optimization()` gère la descente de gradient avec le solveur **Adam** (`torch.optim.Adam`).
- **Epochs** : L'espace est parcouru sur plusieurs pas (epochs). À chaque pas, PyTorch calcule les gradients de l'énergie et des pénalités avec `loss.backward()` puis actualise la position et la rotation des corps rigides.
- **Cycles et Shake (Secousses)** : L'ARN possède un "paysage énergétique" (Energy Landscape) rugueux comportant de vastes *minima locaux* géométriques. L'optimisation est donc découpée en *Cycles*. À la fin d'un cycle, au lieu de s'arrêter dans un cul-de-sac, l'algorithme déclenche l'injection d'un bruit brownien important (`noise_coords`, `noise_angles`) afin de bousculer la structure entière. La force de cette "secousse" diminue au fil des cycles (`decay = 1.0 - cycle/n`). Fait important : si le backbone s'est disloqué (`bb_err`), le bruit temporel est ciblé et amplifié spécifiquement sur les nucléotides disjoints pour les forcer à se re-capter (via `nuc_noise_scale.index_add_`).

---

## 2. Les Optimiseurs "Full-Atom"

### `RNA_DFIRE_Optimizer.py`
Hérite de `RNA_Optimizer` et remplace le terme d'énergie de la "loss" initiale par le calcul du champ DFIRE sur l'entièreté des atomes de la molécule.
- **Vectorisation Massive** : Au lieu d'utiliser des boucles python, les distances de la matrice 2D triangulaire de toutes les paires (`self.pair_i`, `self.pair_j`) sont calculées en une seule opération GPU via `torch.norm`.
- **Interpolation Linéaire** : L'énergie DFIRE est discrète (par "fines tranches" ou "bins" de 0.7 Å). Puisqu'une fonction "en escalier" stalle la dérivée algorithmique (gradient nul !), le programme calcule une interpolation douce : $(1-\alpha) \cdot E_0 + \alpha \cdot E_1$. Ainsi, PyTorch "sent" la pente et guide chaque atome lourdement vers le fond d'un puits énergétique DFIRE.

### `RNA_RASP_Optimizer.py`
Fonctionnement parallèle à DFIRE, adapté à l'architecture complexe du champ **RASP**.
- **Composante de Séparation Séquentielle** : L'énergie d'une paire RASP dépend de la séparation des nucléotides sur la chaîne mère (variable `k` plafonnée de 0 à 5). Le tenseur énergétique se dote donc d'une  4ème dimension (`K_sep x Type1 x Type2 x Bins`).
- **Truncature (Distance Capping)** : Les atomes éloignés au-delà de 20 Å ont une interaction nulle garantie. L’algorithme procède à un "clamp", créant un masque permettant au GPU d'outre-passer leurs énergies afin de soulager le processeur de gradients astronomiques infondés.

---

## 3. Les Optimiseurs "Fast-Track" C3' Uniquement

### `RNA_DFIRE_C3_Optimizer.py` & `RNA_RASP_C3_Optimizer.py`
Traiter toutes les atomes d'une structure génère un tenseur de paires avec un facteur de $O(N_{atomes}^2)$. Sur une modélisation Full-Atom, celà monopolise des GB de capacité Tensor Core GPU.
Ces classes relèvent le défi avec une heuristique extrêmement avantageuse pour de longues séquences :
1. **Actifs vs Inactifs** : Des masques (ex: `mask_c3`) partitionnent la structure. Seuls les atomes cruciaux atterrissent dans les variables requérantes de la RAM : le tenseur DFIRE/RASP n'évalue *plus que* les interactions entre les carbones primordiaux orientant le sucre (`C3'`). De même, seules les paires `O3'` et `P` continuent de figurer dans les équations pour tenir le backbone.
2. **Gain** : On bascule d'une interaction de $\approx 3200 \times 3200$ paires vers un simple subset de $\approx 100 \times 100$ paires C3', menant à une division par cent de l'encombrement spatio-temporel matériel.
3. **Reconstitution Passive (`save_pdb()`)** : Par définition du corps rigide mentionné au chapitre 1, les autres atomes de la base (Adénine, Uracile...) et du sucre sont "ancrés" à leur centre. Bien qu'ignorés par le score énergétique, lorsque l'optimisation trouve la matrice de rotation Euler finale validant l'architecture C3', ces atomes inactifs ont mécaniquement viré, tourné et pivoté de concert. Le code recompose alors leurs coordonnées formelles entières in extremis à la toute dernière ligne permettant l'enregistrement d'un PDB complet rigoureusement exact sur le plan stérique intramoléculaire !
