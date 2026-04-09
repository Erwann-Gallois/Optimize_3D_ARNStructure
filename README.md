# RNA Structure Optimization

This repository provides tools for RNA structure prediction and assessment using statistical potentials. It supports two main optimization models: **Bead-Spring** and **Full-Atom**, with support for **RASP**, **DFIRE-RNA**, and **rsRNASP** potentials.

## 🚀 Features

- **Bead-Spring Optimization**: Efficiently optimize RNA as a chain of beads (e.g., C3') using physics-based constraints (WCA, FENE) and statistical potentials.
- **Full-Atom Optimization**: Refine RNA structures treating each nucleotide as a rigid body with 6 degrees of freedom (rotation and translation).
- **Multiple Potentials**: Integration of **RASP** (All-atom knowledge-based potential), **DFIRE-RNA** (Distance-scaled, Finite Ideal-gas REference), and **rsRNASP** (Residue-specific statistical potential).
- **Advanced Optimizers**: Support for **L-BFGS** and **Adam** optimizers with adaptive noise injection (basin hopping).
- **Visualization**: Tools for monitoring optimization via TensorBoard and generating PDB files for NGLView/PyMOL.

## 📊 Data Sources

- **RASP**: Statistical potential matrices from the [Melo Lab](http://melolab.org/supmat/RNApot/Sup._Data.html).
- **DFIRE-RNA**: Distance-scaled statistical potential for RNA structures.
- **rsRNASP**: Residue-specific statistical potential for RNA structure prediction.

## 🛠 Installation

Recreate the environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate Stage
```

## 💻 Usage

```bash
./launch_opt.sh
```

## 🏗 Project Structure

- `main_bead_springs.py`: Entry point for Bead-Spring optimization.
- `main_full_atom.py`: Entry point for Full-Atom rigid-body optimization.
- `classe/`: Core classes for optimizers and RNA models.
- `potentials/`: Directory containing RASP and DFIRE potential data.
- `docs/`: Detailed technical documentation.

## 📖 Citation

If you use RASP, DFIRE, or rsRNASP potentials, please cite their respective authors:
- **RASP**: Capriotti et al., *Bioinformatics*, 2011. [DOI: 10.1093/bioinformatics/btr093](https://doi.org/10.1093/bioinformatics/btr093)
- **DFIRE-RNA**: Zhang et al., *Journal of Computational Biology*, 2020. [DOI: 10.1089/cmb.2019.0251](https://doi.org/10.1089/cmb.2019.0251)
- **rsRNASP**: Tan et al., *Biophysical Journal*, 2021. [DOI: 10.1016/j.bpj.2021.11.016](https://doi.org/10.1016/j.bpj.2021.11.016)
