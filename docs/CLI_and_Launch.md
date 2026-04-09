# Interfaces and Launching Optimization

The project offers two main entry points depending on the desired optimization model.

## 1. Entry Points (Python)

### `main_bead_springs.py`
Optimization via the **Bead-Spring** model (chain of beads).
```bash
python main_bead_springs.py -s ACGU --score rsRNASP --epochs 100 --cycles 5
```

### `main_full_atom.py`
Optimization via the **Full-Atom** model (rigid bodies).
```bash
python main_full_atom.py -f sequence.fasta --score rasp --lr 0.2
```

## 2. Command Line Arguments (CLI)

Both scripts share common arguments but also have specific options.

### Common Arguments:
- `-s`, `--sequence`: Direct RNA sequence (ex: `ACGU`).
- `-f`, `--fasta`: Path to a FASTA file.
- `--score {rasp, dfire, rsRNASP}`: Statistical scoring function to use (default: `dfire`).
- `-o`, `--output`: Path of the output PDB file.
- `--epochs`: Number of epochs per optimization cycle (default: 50).
- `--cycles`: Number of "basin hopping" cycles (default: 20).
- `--lr`: Learning rate (default: 0.2).
- `--noise-coords`: Intensity of the noise added to coordinates at each cycle (default: 0.5).
- `-v`, `--verbose`: Enables detailed display during optimization.
- `--cif`: Export the final structure in .cif format.

### Specific to Bead-Spring:
- `--k`: Spring stiffness constant (default: 40.0).
- `--l0`: Equilibrium distance between beads (default: 5.5).
- `--bead-atom`: Atom used as the center of the bead (default: `C3'`).

### Specific to Full-Atom:
- `--backbone-weight`: Weight of the penalty for backbone connectivity (default: 100).
- `--noise-angles`: Intensity of the noise added to rotation angles at each cycle (default: 0.2).

## 3. Launch Scripts (Bash)

To simplify execution without memorizing all arguments, several assistant scripts are provided:
- **`launch_bead_springs.sh`**: Prepares an optimized run for the bead model.
- **`launch_full_atom.sh`**: Prepares an optimized run for the full-atom model.
- **`launch_all_scores_bead_springs.sh`**: Sequentially runs optimizations with all scores for the bead model.
- **`launch_all_scores_full_atom.sh`**: Sequentially runs optimizations with all scores for the full-atom model.
- **`launch_opt.sh`**: Interactive script to choose the model and scoring function.

These scripts automatically manage the creation of `fichier_arn/` and `resultat/` folders if necessary.
