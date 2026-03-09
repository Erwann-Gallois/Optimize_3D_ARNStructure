# RNA Structure Optimization with RASP

This repository provides tools for RNA structure prediction and assessment based on the **RASP** (All-atom knowledge-based potential) framework. It includes implementations for both all-atom gradient descent and rigid-body optimization of RNA structures.

## 🚀 Features

- **All-Atom Gradient Descent**: Optimize RNA structures by minimizing the RASP potential using gradient information.
- **Rigid-Body Optimization**: A nucleotide-by-nucleotide approach to structural refinement, treating each nucleotide as a rigid entity.
- **Potential Parsing**: Tools to parse and utilize statistical potential matrices.
- **Visualization**: Integration for viewing RNA structures and monitoring optimization via TensorBoard.

## 📊 Data Source

The statistical potential matrices used in this project are sourced from the **Melo Lab** website:
[http://melolab.org/supmat/RNApot/Sup._Data.html](http://melolab.org/supmat/RNApot/Sup._Data.html)

## 📖 Citation

If you use this code or the RASP potentials in your research, please cite the following paper:

> Emidio Capriotti, Tomas Norambuena, Marc A. Marti-Renom, Francisco Melo, **All-atom knowledge-based potential for RNA structure prediction and assessment**, *Bioinformatics*, Volume 27, Issue 8, April 2011, Pages 1086–1093, [https://doi.org/10.1093/bioinformatics/btr093](https://doi.org/10.1093/bioinformatics/btr093)

## 🛠 Installation

You can recreate the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate Stage
```

## 💻 Usage

### All-Atom Gradient Optimization
Run the standard optimization script:
```bash
python main.py
```

### Rigid-Body Optimization
Run the nucleotide-by-nucleotide rigid optimization:
```bash
python main_rigid.py
```

## 🏗 Project Structure

- `RNA_RASP_Gradient.py`: Core logic for all-atom gradient-based optimization.
- `RNA_RASP_Rigid.py`: Core logic for rigid-body nucleotide optimization.
- `parse_rasp_potentials.py`: Utilities for handling RASP potential files.
- `main.py` / `main_rigid.py`: Entry points for running the optimizations.
- `potentials/`: Directory containing the RASP potential matrices.
