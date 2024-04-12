import sys
sys.path.append("..")
from src import train_utils

def train(model_config = {
                    "model_type": "traditional", 
                    "clf": "HistGradientBoostingClassifier",
                    "name": "HGBC_run1_30sec"
                    },
          data_config = {
                    "location": "../../data/genres_original", 
                   "use_file": "../../data/features_30_sec.csv", # so no preprocess
                   "split_shuffle": True,
                   "split_stratify": True,
                   "train_split": 0.7, 
                   "test_split": 0.15,
                   "seed": 2024, 
                   },
          train_config = {
                      "model_type": "traditional",
                    "grid_search": {}, #'max_leaf_nodes': [30, 60, 120]},
                    "n_folds": 5, 
                    "metric": ['f1_macro', 'accuracy'], 
                    "seed": 2024,
                    "save_location": "../output"
                    },
         verbose = False):
    
    model = train_utils.get_model(**model_config)
    if verbose:
        print("Model obtained")

    train, val, test, cache = train_utils.get_data(**data_config)
    data = [train, val, test]
    if verbose:
        print("Data preprocessed and split")
        print("Starting training")
    model = train_utils.train(model, data, data_cache= cache, **train_config)

    if verbose:
        print("Trained and outputs saved")
    return model

if __name__ == "__main__":
    train()