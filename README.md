# Music Genre Classification - Project Overview

This repository contains the source code and related model implementations for classifying Spotify music genres using an aggregated/matched Spotify Web API dataset. It contains preprocessing, feature extraction, and exploration notebooks plus utility scripts used for modeling.

## Repository Overview

- `datasets/`: raw CSV datasets (ex: `full_spotify_dataset.csv`).
- `models/`: our implementations for the chosen models (K Nearest Neighbors, Random Forest, Logistic Regression, Gradient Boost )
- `preprocess/`: Jupyter notebooks and output for preprocessing and feature extraction:
  - `preprocess/preprocessing.ipynb` - main file that preprocesses the given data, and outputs a cleaned dataset for other models to use
  - `explore.ipynb` - EDA visualizations.
  - `output/`: the outputs artifacts including the full final dataset `spotify_preprocessed.csv`
  - `util/`: helper scripts used by notebooks and experiments:
  - `util/classify_genre.py` - utility functions used for syntax genre grouping
- `results/`: some of the figures

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
To run the models, make sure the preprocessed dataset exist at `../preprocess/output/spotify_preprocessed.csv`. If not, update this line in the code: `df = pd.read_csv("path/to/your/spotify_preprocessed.csv")`

- To run the KNN model, open `models/KNN.py`, and run the file. Best parameters, and performance metrics for all groups and overall performance will be printed.
- To run the RF model, open `models/RF.py`, and run the file. Best parameters, and performance metrics for all groups and overall performance will be printed.
- To simulate the logistic regression model, open `models/logistic-model.py`. In the main function, two functions are commented out. Uncommenting the function `showScore(scaled_X_train, y_train, scaled_X_test, y_test)` will run the model on the found best hyperparameters, iterating through the C values, and displaying a graph of its precision, accuracy, and balanced accuracy. Uncommenting the function `grid_lr(scaled_X_train, y_train, scaled_X_test, y_test, 8)` will run the gridsearch. The last positional argument represents the number of jobs the grid search will use. Change appropriately.
- To run the Gradient Boost model, open `models/gradient-boost.py`, and run the file. The code will perform feature encoding, train-test split, train 3 models (using softprob, softmax, custom focal loss), run hyperparameter tuning (gridSearch CV), print the best parameters, evaluation metrics, and display evaluation bar chart.

## Notes and pointers

- Used plots are included in the project report but may also be present in the `output` directories
- There are no trained model files in the repository by default; save any produced models into a new `models/` directory if you add them.
