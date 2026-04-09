# Parsing Statistical Potentials (Energies)

The optimization algorithms use **statistical force fields** (empirical potentials). These scores seek to reproduce the foldings seen in nature by observing the geometry of known structures in the PDB.

The project integrates the potential matrices **DFIRE-RNA**, **RASP**, and **rsRNASP**.

### 1. `parse_dfire_potentials.py`
The DFIRE module calculates interaction energies based on a distance normalized by a reference state (finite-ideal-gas).
- **`load_dfire_potentials()`**: Loads the `matrice_dfire.dat` matrix from the `potentials/` folder.
- **Atom Types**: DFIRE defines precise types combining the residue name (A, U, C, G) and the atom name (e.g., `G_N1`, `A_C8`).
- **Interpolation**: To make the potential differentiable (essential for gradient descent), the code uses **linear interpolation** between distance bins (0.5 Å). This allows PyTorch to calculate a continuous gradient even between two discrete values of the matrix.

### 2. `parse_rasp_potentials.py`
The RASP potential is more complex and takes into account the sequential separation between residues.
- **`load_rasp_potentials()`**: Loads RASP energy files and builds a 4D tensor: `[separation_k, type_A, type_B, distance]`.
- **Sequential Separation**: The energy varies depending on whether the nucleotides are close or distant in the sequence (6 categories from 0 to >4).
- **RASP Types**: Uses 23 atom types specified by the original model.
- **Modularity**: The parser supports both "Full-Atom" mode (all types) and "C3'" mode (only the bead atom), allowing its use in both optimization models.

### 3. `parse_rsrnasp_potentials.py`
The **rsRNASP** (residue-specific statistical potential) is a recent approach (2021) that uses a fine discretization of atomic geometry.
- **`load_rsrnasp_potentials()`**: Loads the `short-ranged.potential` and `long-ranged.potential` files.
- **Atom Types**: rsRNASP defines **85 unique atom types** (22 for A, 20 for U, 20 for C, 23 for G), covering all heavy atoms of the structure.
- **Sequential Reach**: Unlike RASP which has 6 categories, rsRNASP simplifies into two states:
    - **Short-range**: $|i-j| \le 4$
    - **Long-range**: $|i-j| \ge 5$
- **Performance**: Calculations are optimized for PyTorch tensors, allowing rapid evaluation despite the high number of atom types.

### 4. Integration into Optimizers
In the optimization classes (folder `classe/`), these potentials are converted into **PyTorch Tensors**.
- Distance calculations are vectorized to be executed massively on GPU.
- **Capping** techniques (truncation) are applied to ignore interactions beyond a certain distance (typically 20 Å for RASP/DFIRE, 24 Å for rsRNASP), optimizing performance.
