from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import tensorflow as tf


import numpy as np
import pandas as pd
import joblib
import pickle 
import os

import sys
sys.path.append("..")
from src import classifiers
from src import evaluate
from src import preprocessing

LABEL_DICTIONARY = preprocessing.LABEL_DICTIONARY
SAMPLE_RATE = preprocessing.SAMPLE_RATE

def get_model(name, 
              model_type, 
              clf):
    if model_type == "traditional":
        model = classifiers.TraditionalModel(name, clf)
    else:
        model = classifiers.CNN(name)
    return model

def get_data(location = "../../data/genres_original",
             use_file = None,
             num_segments = 10, 
             split_shuffle = True, 
             split_stratify = True,
             train_split = 0.7,
             test_split = 0.15,
             seed = 2024,
             batch_size = 16, 
             epochs = 10,) -> None:
    
    cache = {}
    val_split = 1-(train_split+test_split)
    if val_split<0:
        raise Exception("train_split+test_split provided is >1")

    if use_file is not None:
        # just use features
        data = pd.read_csv(use_file)
        if "filename" in data.columns:
            data = data.drop(columns = "filename")
        labels = list(np.unique(data['label']))
        ids = np.arange(0, len(labels))
        data['label'] = data['label'].apply(lambda x: labels.index(x))
        y = data['label']
        X = data.drop(columns = "label")
        cache['le'] = labels
        cache['features'] = 'standard'
        
        # train-test-split
        
        stratify = y if split_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split,
                                                        shuffle = split_shuffle, stratify=stratify,
                                                        random_state=seed)

        stratify = y_train if split_stratify else None
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split,
                                                        shuffle = split_shuffle, stratify=stratify,
                                                        random_state=seed)

        train_data = {"X": X_train, "y": y_train}
        val_data = {"X": X_val, "y": y_val}
        test_data = {"X": X_test, "y": y_test}

    else:
        # do the preprocess using audio for CNN        
        AUTOTUNE = tf.data.AUTOTUNE

        file_paths = [file for file in glob.glob(f"{location}/**/*.wav", recursive = True)]
        labels = [i.split("/")[-2] for i in file_paths]

        stratify = labels if split_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=test_split,
                                                        shuffle = split_shuffle, stratify=stratify,
                                                        random_state=seed)

        stratify = y_train if split_stratify else None
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split,
                                                        shuffle = split_shuffle, stratify=stratify,
                                                        random_state=seed)

        train_data = tf.data.Dataset.from_tensor_slices(X_train)
        NUM_SAMPLES = train_data.cardinality().numpy()
        
        val_data = tf.data.Dataset.from_tensor_slices(X_val)
        test_data = tf.data.Dataset.from_tensor_slices(X_test)

        train_data = train_data.map(preprocessing.parse_function)
        val_data = val_data.map(preprocessing.parse_function)
        test_data = test_data.map(preprocessing.parse_function)
        
        

        train_data = train_data.batch(batch_size, drop_remainder= True)
        val_data = val_data.batch(batch_size, drop_remainder= True)
        test_data = test_data.batch(batch_size, drop_remainder= True)

        train_data = train_data.prefetch(buffer_size = AUTOTUNE).cache().repeat(epochs)
        val_data = val_data.prefetch(buffer_size = AUTOTUNE).cache().repeat(epochs)
        test_data = test_data.prefetch(buffer_size = AUTOTUNE).cache().repeat(epochs)
        cache['features'] = 'melspectogram'
        cache['EPOCHS'] = epochs
        cache['STEPS_PER_EPOCH'] = int(NUM_SAMPLES/batch_size)
        

    return train_data, val_data, test_data, cache


def train(model, 
          data, 
          model_type = "traditional",
          data_cache = {},
          grid_search = {},
          n_folds = None, 
          metric = ['f1', 'accuracy'], 
          seed = 2024,
          save_location= "../output",):
    
    np.random.seed = seed
    
    if model_type == "traditional":
        if n_folds is not None:
             # cross validation
             X = pd.concat([data[0]['X'], data[1]['X']], axis = 0)
             y = pd.concat([data[0]['y'], data[1]['y']], axis = 0)

             X_test = data[2]['X']
             y_test = data[2]['y']

        else:
            X, y = data[0]['X'], data[0]['y']
            X_test = data[1]['X']
            y_test = data[1]['y']
    else:
        X = data[0]
        y = None
        X_test = data[2]
        y_test = None
    
    history = model.fit(X, y, n_folds = n_folds, grid_search = grid_search, metric = metric, **data_cache,
                       save_location = save_location, val = data[1])
    
    if model_type == "traditional":
        # save model
        if len(save_location)>0:
            folder = f"{save_location}/{model.name}"
            os.makedirs(folder, exist_ok=True)
            joblib.dump(model.classifier, f"{folder}/classifier.joblib")
            if model.search is not None:
                # save cv_results
                with open(f'{folder}/cv_results.pkl', 'wb') as f:
                    pickle.dump(model.search.cv_results_, f)

            if len(data_cache)>0:
                with open(f'{folder}/data_cache.pkl', 'wb') as f:
                    pickle.dump(data_cache, f)
    
    # evaluate --> if val data available (i.e. no CV), evaluate on that, else use test data
    y_pred = model.predict(X_test)
    lis = np.unique(y_test)
    if "le" in data_cache:
        lis = [data_cache['le'][i] for i in lis]
    results = evaluate.classification_metrics(y_test, y_pred, lis = lis, location=folder)
    model.results = results
    
    # save test-results
    if len(save_location)>0:               
        with open(f'{folder}/test_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    return model