# HDFT Paper

This directory contains material for the hydrodynamic fluctuation diffusion theory (HDFT) paper.  The contents are organized into the following subfolders:

- **notebooks/** – Jupyter notebooks used for analysis and figure generation.
- **scripts/** – Python scripts for running standalone simulations.
- **results/** – Output files produced by the notebooks or scripts, including training logs and animations.
- **run_info/** – Pickled objects describing each simulation run (metrics, history, predictions).

## Usage

Open the notebooks with JupyterLab or your preferred notebook interface:

```bash
jupyter lab notebooks/5_22_24.ipynb
```

To run the simulation script from the command line:

```bash
python scripts/simulation_2023_06_12.py
```

The script and notebooks rely on standard scientific Python packages such as `numpy`, `matplotlib`, and `scipy`.
