# Utilitaires et Interface avec AmberTools (`fonction.py`)

Le fichier `fonction.py` sert de boite à outils (Wrapper/Helpers). Toutes les étapes précédant physiquement l'optimisation PyTorch sont réalisées ici.

## Préparation de la séquence : La génération (AmberTools - tleap)
Étant donné une séquence primaire textuelle (Ex: `AAGCU`), il faut un squelette 3D canonique et rigide pur à soumettre à notre optimiseur C3'/Full-Atom.
- `generer_arn_droit(sequence)` : Utilise les APIs logicielles du standard académique de bio-molécule dynamique **AmberTools** (logiciel indispensable en dépendance de la machine de test, souvent géré par environnement virtuel Conda `Stage`).
  - La fonction écrit dynamiquement un script d'automatisation nommé `instructions_tleap.in` avec les paramètres du système Amber (champ de forces `leaprc.RNA.OL3`). 
  - La structure longue est fragmentée de 50 en 50 si nécessaire et insérée via le module CLI en sous-processus python (`subprocess.run()`).
  - L'output 3D généré est enregistré et `leap.log` ainsi que le fichier temporaire de script sont écrasés (Clean-up de tleap).
- `enlever_hydrogene()` : RASP et DFIRE ne prennent traditionnellement pas en paramètre les hydrogènes, et les simuler pendant la descente de gradiants multiplierait considérablement par trois la perte mémoire (RAM/GPU). On filtre le `.pdb` Amber avec un masque logique Pandas.

## Nettoyage des Artefacts
- `fix_amber_pdb()` : `tleap` produit de temps en temps des erreurs de fin de chaine d'acide nucléique, ne générant notamment pas d'atomes critiques terminaux du squelette phosphodiester (`OP3`). La fonction lit, avec la bibliothèque lourde structurelle **BioPython**, toute arborescence du `PDB`, parse et recrée si besoin un noeud géométrique pour l'atome en décalage de son frère `OP2`. Ce contournement évite de casser l'architecture paramétrique par atome manquant. 

## Visualisation
- `view_structure()` : Raccourci vers [NglView](https://github.com/nglviewer/nglview), particulièrement utile sur des environnements iPython (Notebook Python) tels que `test_alphafold.ipynb`. Elle paramètre l'affichage de "bâton" pour le squelette complet (`licorice`) et applique une surbrillance de type `spacefill` couleur `orange` directement focalisée sur les atomes `_P` (Le phosphore, véritable "noeud vertébral" du backbone ARN).

## Workflow Initial complet
- `generer_first_structure()` est simplement la consolidation encapsulante de `generer_arn_droit` $\rightarrow$ `enlever_hydrogene` $\rightarrow$ `fix_amber_pdb()`. 
Les fichiers passés par cette pipeline atterrissent historiquement dans le dossier local `/fichier_arn/`.
