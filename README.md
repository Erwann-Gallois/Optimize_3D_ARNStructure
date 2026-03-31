# RNA Structure Optimization

This repository provides tools for RNA structure prediction and assessment using statistical potentials. It supports two main optimization models: **Bead-Spring** and **Full-Atom**, with support for **RASP** and **DFIRE-RNA** potentials.

## 🚀 Features

- **Bead-Spring Optimization**: Efficiently optimize RNA as a chain of beads (e.g., C3') using physics-based constraints (WCA, FENE) and statistical potentials.
- **Full-Atom Optimization**: Refine RNA structures treating each nucleotide as a rigid body with 6 degrees of freedom (rotation and translation).
- **Multiple Potentials**: Integration of both **RASP** (All-atom knowledge-based potential) and **DFIRE-RNA** (Distance-scaled, Finite Ideal-gas REference).
- **Advanced Optimizers**: Support for **L-BFGS** and **Adam** optimizers with adaptive noise injection (basin hopping).
- **Visualization**: Tools for monitoring optimization via TensorBoard and generating PDB files for NGLView/PyMOL.

## 📊 Data Sources

- **RASP**: Statistical potential matrices from the [Melo Lab](http://melolab.org/supmat/RNApot/Sup._Data.html).
- **DFIRE-RNA**: Distance-scaled statistical potential for RNA structures.

## 🛠 Installation

Recreate the environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate Stage
```

## 💻 Usage

### 1. Bead-Spring Optimization
Optimize a simplified representation of RNA:
```bash
python main_bead_springs.py --sequence GGGAAACCC --score dfire --epochs 50 --cycles 10
```

### 2. Full-Atom Optimization
Refine all atoms using rigid-body transformations:
```bash
python main_full_atom.py --fasta example.fasta --score rasp --lr 0.1
```

### 3. Launch Scripts
Use the provided shell scripts for standard runs:
```bash
./launch_bead_springs.sh
./launch_full_atom.sh
```

## 🏗 Project Structure

- `main_bead_springs.py`: Entry point for Bead-Spring optimization.
- `main_full_atom.py`: Entry point for Full-Atom rigid-body optimization.
- `classe/`: Core classes for optimizers and RNA models.
- `potentials/`: Directory containing RASP and DFIRE potential data.
- `docs/`: Detailed technical documentation.

## 📖 Citation

If you use RASP or DFIRE potentials, please cite their respective authors:
- **RASP**: Capriotti et al., *Bioinformatics*, 2011. [DOI: 10.1093/bioinformatics/btr093](https://doi.org/10.1093/bioinformatics/btr093)
- **DFIRE-RNA**: (Link/Citation for DFIRE-RNA if available)
