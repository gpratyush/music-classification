import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score

def classification_metrics(y_true, y_pred, lis, location):
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels = lis)
    cmd.plot()
    plt.savefig(f"{location}/cm.png")
    print(classification_report(y_true, y_pred, target_names=lis))
    return accuracy
