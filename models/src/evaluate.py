import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
import numpy as np
import pandas as pd


import sys
sys.path.append("../..")
from models.src import utils

def classification_metrics(y_true, y_pred, lis, location="", verbose = False):
    lis = [str(i) for i in lis]
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels = lis)
    cmd.plot()
    if len(location)>0:
        plt.savefig(f"{location}/cm.png")
    plt.close()
    report = classification_report(y_true, y_pred, target_names=lis, output_dict = True)
    perclass = lambda x: np.array([report[i]['f1-score'] for i in x])
    # print(report)
    results = {'accuracy': report['accuracy'],
              'f1-macro': report['macro avg']['f1-score'],
              'worst-performing-label': lis[np.argmin(perclass(lis))],
              'best-performing-label': lis[np.argmax(perclass(lis))]}
    return results

def cv_results_evaluate(cv_results):
    """ list of cv_results """
    if type(cv_results)!=list:
        cv_results= [cv_results]
    cv_results = pd.DataFrame([i for i in cv_results]).reset_index()
    cv_results = utils.explode(cv_results,list(set(cv_results.columns)-{'index'}) )
    return cv_results