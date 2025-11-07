# TRCNO — Kernel solvers & PINN notebooks

This repository contains research/experiment Jupyter notebooks and a shared numerical kernel solver module used across them.

Summary
- kernel_solvers.py: central implementation of the numerical kernel solvers (solver_2x2 and K_solver_2x2). Notebooks now import the solver functions from this module instead of defining them inline.
- Several Jupyter notebooks (.ipynb) for experiments, data generation and PINN/DeepONet training.

Repository layout (relevant files)
- `kernel_solvers.py`  — shared solver implementations (solver_2x2, K_solver_2x2).
- `*.ipynb`  — multiple notebooks (example: `arz-no-mu2k.ipynb`, `arz-no-mu2control.ipynb`, `data_gene_train.ipynb`, `arz-pinn-mu2k.ipynb`, ...).

Quickstart
1. Clone the repo and change to the repository root (so local imports like `utilities3`, `models` work):

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Start Jupyter (Lab or Notebook) and open the notebooks:

   ```bash
   jupyter lab
   ```

Design notes
- The numerical kernel solver code that used to appear inline in multiple notebooks has been extracted to `kernel_solvers.py` for maintainability and reuse. Notebooks call:

  - `from kernel_solvers import solver_2x2, K_solver_2x2`

- Many notebooks already contain a small comment cell indicating the functions were moved out and an import line has been inserted in the top import cell.

Example (Python snippet)

```python
# from project root
from kernel_solvers import K_solver_2x2

# `fun` should provide callables: mu(x), lam(x), lam_d(x), mu_d(x), c_1(x), c_2(x), and scalar q
# N_g is the grid size used by the solver
# Example usage (adapt to your notebook objects):
# Kvu, Kvv = K_solver_2x2(fun, N_g)
```

Notes & caveats
- Some notebooks import local helper modules (for example `utilities3`, `models`) that live in the repository; run Jupyter from the repo root so local imports resolve.
- I intentionally preserved the dynamic `Struct`-style parameter objects used across notebooks for compatibility.
- This README and `requirements.txt` provide a minimal environment to run the notebooks, but exact versions may be updated based on your target platform (CPU vs GPU) and DeepXDE backend choices.

How to validate quickly
- From the repo root, with the virtualenv activated, you can run a short import check in a Python REPL:

  ```python
  from kernel_solvers import solver_2x2, K_solver_2x2
  print('kernel_solvers import OK')
  ```

If that runs without ImportError, notebooks should be able to import the module when started from the same folder.

Contributing
- If you update `kernel_solvers.py`, please keep backward compatibility for the simple wrapper `K_solver_2x2(fun, N_g)` used by notebooks.

License
- MIT (adjust if you want a different license)

Contact
- Add your contact or project link here before uploading to GitHub.
