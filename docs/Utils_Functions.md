# Utilities and Interface with AmberTools (`fonction.py`)

The `fonction.py` file serves as a toolbox (Wrapper/Helpers). All steps physically preceding the PyTorch optimization are performed here.

## Sequence Preparation: Generation (AmberTools - `tleap`)
Given a primary textual sequence (Ex: `AAGCU`), a canonical and rigid 3D backbone is needed to be submitted to our C3'/Full-Atom optimizer.
- `generer_arn_droit(sequence)`: Uses the standard academic dynamic bio-molecule software APIs **AmberTools** (indispensable dependency, often managed by the Conda `Stage` environment).
  - The function dynamically writes an automation script named `instructions_tleap.in` with the Amber system parameters (force field `leaprc.RNA.OL3`).
  - Longer structures are fragmented every 50 residues if necessary and inserted via the CLI module in a Python subprocess (`subprocess.run()`).
  - The generated 3D output is saved, and `leap.log` as well as the temporary script file are overwritten (AmberTools clean-up).
- `enlever_hydrogene()`: RASP and DFIRE traditionally do not take hydrogens as input, and simulating them during gradient descent would significantly increase memory consumption (RAM/GPU). The Amber `.pdb` is filtered using a logical Pandas mask.

## Artifact Cleanup
- `fix_amber_pdb()`: `tleap` occasionally produces end-of-chain artifacts in nucleic acids, specifically failing to generate critical terminal phosphodiester backbone atoms (`OP3`). The function uses the **BioPython** structural library to parse the `PDB` tree and recreate a geometric node for the atom if missing, using the neighboring `OP2` as a reference. This workaround prevents the parametric architecture from breaking due to missing atoms.

## Visualization
- `view_structure()`: Shortcut to [NglView](https://github.com/nglviewer/nglview), particularly useful in iPython environments (Jupyter Notebooks) such as `test_alphafold.ipynb`. It configures the display to show the full skeleton as sticks (`licorice`) and applies a `spacefill` highlight in `orange` focused on the `P` atoms (Phosphorus, the vertebral node of the RNA backbone).

## Full Initial Workflow
- `generer_first_structure()` is simply the consolidating wrapper of `generer_arn_droit` $\rightarrow$ `enlever_hydrogene` $\rightarrow$ `fix_amber_pdb()`.
Files passed through this pipeline historically land in the local `/fichier_arn/` folder.
