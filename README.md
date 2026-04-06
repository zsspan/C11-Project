# Music Genre Classification - Project Overview

This repository contains the source code and related model implementations for classifying Spotify music genres using an aggregated/matched Spotify Web API dataset. It contains preprocessing, feature extraction, and exploration notebooks plus utility scripts used for modeling.

## Repository Overview
- `datasets/`: raw CSV datasets (ex: `full_spotify_dataset.csv`).
- `preprocess/`: Jupyter notebooks and output for preprocessing and feature extraction:
  - `preprocess/preprocessing.ipynb` - main file that preprocesses the given data, and outputs a cleaned dataset for other models to use
  - `explore.ipynb` - EDA visualizations.
  - `output/`: the outputs artifacts including the full final dataset `spotify_preprocessed.csv`
- `util/`: helper scripts used by notebooks and experiments:
  - `util/classify_genre.py` - utility functions used for syntax genre grouping

## Running the project
1. Create and activate a Python environment and then install dependencies (Ex: using PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Preprocess the data (already done):
- Open `preprocess/preprocessing.ipynb` and run the cells. The notebook used the original dataset but already outputted the full dataset used for the given models

3. Explore / visualize:
- Open `preprocess/explore.ipynb` and run cells. It reads `spotify_preprocessed.csv` and produces plots (heatmap, pairplot, genre distribution).

4. Models
- TODO
- Add how to run models

## Notes and pointers
- Used plots are included in the project report but may also be present in the `output` directories
- There are no trained model files in the repository by default; save any produced models into a new `models/` directory if you add them.
