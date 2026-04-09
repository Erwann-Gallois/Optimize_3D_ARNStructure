# The Optimizers (Folder `classe/`)

The project is built around two distinct optimization philosophies, both implemented with PyTorch for tensor calculations and autodifferentiation.

## 1. Bead-Spring Model: `BeadSpringOptimizer.py`

Unlike an all-atom approach, this model reduces each nucleotide to a single bead (generally the **C3'** atom). It's an extremely fast mesoscopic physics model.

### A. Cohesion Potentials (Physics)
To maintain the integrity of the RNA chain without a full atomic structure, two classical forces are used:
1. **WCA (Weeks-Chandler-Andersen)**: A purely repulsive version of the Lennard-Jones potential. It prevents beads from interpenetrating (excluded volume).
2. **FENE (Finitely Extensible Nonlinear Elastic)**: A non-linear spring that connects beads together. It prevents the chain from breaking while limiting the maximal extension of bonds.

### B. Statistical Potential
The total energy is completed by a score from **RASP**, **DFIRE-RNA**, or **rsRNASP**, calculated solely on distances between the selected beads.
- **Derived Classes**: `BeadSpringRASPOptimizer`, `BeadSpringDFIREOptimizer`, and `BeadSpringRsRNASPOptimizer`.

---

## 2. Rigid Body Model: `FullAtomOptimizer.py`

This approach treats each nucleotide as a rigid, undeformable entity. It's the preferred approach for high-precision refinement.

### A. Geometric Parametrization (6 DOF)
Each nucleotide has 6 optimizable degrees of freedom:
- **Translation (3D)**: Position of the reference center (`ref_coords`).
- **Rotation (3D)**: Spatial orientation via Euler angles (`rot_angles`).

The position of each atom is reconstructed by: $Coords = Ref\_Coords + R \times Offsets$, where offsets are the fixed internal relative positions of the nucleotide's atoms.

### B. Backbone Constraints
Since nucleotides are separate rigid bodies, a "backbone" penalty is applied to force the **O3'** atom of nucleotide $i$ to remain close to the **P** atom of nucleotide $i+1$ (target distance ~1.6 Å).

---

## 3. Optimization Algorithms

The project supports several descent algorithms:
- **Adam**: Robust for large oscillations and exploratory phases.
- **L-BFGS**: Very efficient quasi-Newton algorithm for precise convergence to the local minimum once the structure is globally correct.

### Basin Hopping (Shake)
To escape from local minima, the optimization is segmented into **Cycles**. At each new cycle, a controlled noise (shake) is injected into the coordinates and angles to explore new configurations.

---

## 4. Score Integration (RASP, DFIRE & rsRNASP)

The final classes (e.g., `FullAtomRsRNASPOptimizer`) combine these physical models with the potential matrices. They use **linear interpolation** to make discrete energy grids (bins) differentiable, allowing PyTorch to calculate the gradients necessary to move the nucleotides.
