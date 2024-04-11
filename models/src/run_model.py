import sys
sys.path.append("..")
sys.path.append("../..")

from models.src import run_utils
import pickle

location = "models/output/HGBC_run1_30sec"
model_location = f"{location}/classifier.joblib"
data_cache_location = f"{location}/data_cache.pkl"

def run(file):
    
    model = run_utils.load_model(model_location)
    X = run_utils.get_data(file)
    with open(data_cache_location, 'rb') as f:
        data_cache = pickle.load(f)
    y_pred = run_utils.run(model, X, data_cache)
    return [y_pred[0]]*3


if __name__ == "__main__":
    """ take the audio file as numpy array and
    runs the model to top 3 predictions ordered in a list
    """
    print(run("../../data/genres_original/blues/blues.00000.wav"))
