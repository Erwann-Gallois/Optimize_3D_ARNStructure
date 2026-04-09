# Wiki: RNA Structure Optimization

Welcome to the **RNA Structure Optimization** project documentation. This project enables the generation and optimization of 3D RNA structures from their primary sequences using statistical potentials such as **DFIRE-RNA**, **RASP**, and **rsRNASP**.

The application offers two distinct approaches:
1. **Bead-Spring Model**: A simplified representation where each nucleotide is reduced to a "bead" (typically the C3' atom), optimized using physical constraints (WCA, FENE).
2. **Full-Atom Model**: A complete representation where each nucleotide is treated as a rigid body (6 degrees of freedom) for high-precision optimization of all heavy atoms.

## Project Structure

Here are the key components and code files present in the repository:

### 1. Optimization Core (Folder `classe/`)
The optimization logic leverages PyTorch to accelerate geometric calculations and support gradient descent (**Adam**) and quasi-Newton (**L-BFGS**) algorithms.
- **`BeadSpringOptimizer.py`**: Handles bead-based optimization with cohesion forces and statistical potentials.
- **`FullAtomOptimizer.py`**: Manages rigid-body optimization for all heavy atoms.
- **`BeadSpringRASPOptimizer.py`** / **`FullAtomDFIREOptimizer.py`** (etc.): Specific implementations combining a model with a scoring function.

### 2. User Interfaces (CLI & Scripts)
- **`main_bead_springs.py`**: Main entry point for the bead-spring optimization model.
- **`main_full_atom.py`**: Main entry point for the full-atom rigid-body optimization model.
- **`launch_bead_springs.sh`** / **`launch_full_atom.sh`**: Bash scripts to simplify execution with optimized default parameters.

### 3. Potential Parsing
Statistical potentials are managed by:
- **`parse_dfire_potentials.py`**: Loads and processes the DFIRE-RNA matrix.
- **`parse_rasp_potentials.py`**: Loads and processes RASP energy files.
- **`parse_rsRNASP_potentials.py`**: Loads and processes rsRNASP energy files.

### 4. Utility Functions
- **`fonction.py`**: Groups functions for initial structure preparation (via AmberTools' `tleap`), PDB cleanup, and FASTA file reading.

---
For more details, see the following pages:
- [Optimizers (Folder `classe/`)](Optimizers.md)
- [Interfaces and Execution (CLI & Bash)](CLI_and_Launch.md)
- [Potential Parsing (DFIRE, RASP & rsRNASP)](Potentials_Parsing.md)
- [Utility Functions and AmberTools](Utils_Functions.md)
