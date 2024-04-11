import joblib
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
sys.path.append("../..")


from models.src import classifiers
from models.src import preprocessing

def load_model(location):
    name = location.split("/")[-2]
    clf = joblib.load(location)
    return classifiers.TraditionalModel(name = name, classifier=clf)

def get_data(file, data_cache = {}, kind = "encode_audio"):
    if kind=="encode_audio":
        # need to process to create features
        segments = preprocessing.encode_audio(file, n_segments=1)
        X = pd.DataFrame(segments)
    else:
        pass
    return X

def run(model, X, data_cache = {}):
    y_pred = model.predict(X)
    if "le" in data_cache:
        y_pred = [data_cache['le'][i] for i in y_pred]
    return y_pred
