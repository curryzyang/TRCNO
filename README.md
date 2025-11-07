# Operator Learning for Traffic Congestion

This repository contains research/experiment Jupyter notebooks and a shared numerical kernel solver module used across them.

Repository: https://github.com/curryzyang/TRCNO

Associated paper: https://www.sciencedirect.com/science/article/pii/S0968090X24004492

Repository layout (relevant files)
- `kernel_solvers.py`  — shared solver implementations (solver_2x2, K_solver_2x2).
- `*.ipynb`  — multiple notebooks (example: `arz-no-mu2k.ipynb`, `arz-no-mu2control.ipynb`, `data_gene_train.ipynb`, `arz-pinn-mu2k.ipynb`, ...).

Quickstart
1. Clone the repo and change to the repository root (so local imports like `utilities3`, `models` work):

   ```bash
   git clone https://github.com/curryzyang/TRCNO.git
   cd TRCNO
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

License
- MIT

Contact
- Paper: https://www.sciencedirect.com/science/article/pii/S0968090X24004492
- Repository: https://github.com/curryzyang/TRCNO
- Email: yzhang169@connect.hkust-gz.edu.cn

Citation (BibTeX)

```bibtex
@article{zhang2025mitigating,
   title={Mitigating stop-and-go traffic congestion with operator learning},
   author={Zhang, Yihuai and Zhong, Ruiguo and Yu, Huan},
   journal={Transportation Research Part C: Emerging Technologies},
   volume={170},
   pages={104928},
   year={2025},
   publisher={Elsevier}
}
```
