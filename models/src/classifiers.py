from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class TraditionalModel:
    def __init__(self, name, classifier) -> None:
        self.name = name
        model_match = {"HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
                       "AdaBoostClassifier": AdaBoostClassifier(),
                       "RandomForestClassifier": RandomForestClassifier(),
                       "GaussianNB": GaussianNB(),
                       "KNeighborsClassifier": KNeighborsClassifier(),
                       "MLPClassifier": MLPClassifier(),
                       "SVC": SVC(),
                       "DecisionTreeClassifier": DecisionTreeClassifier(),
                      }        
        if type(classifier) == str:
            self.classifier = model_match[classifier]
        else:
            self.classifier = classifier
        self.search = None
    
    def fit(self, X, y, n_folds = None, grid_search = {}, metric = None, **kwargs):
        if n_folds is not None:
            # CV
            search = GridSearchCV(self.classifier, grid_search,
                              scoring = metric, refit='accuracy', cv = n_folds)
            search.fit(X, y)
            self.search = search
            self.classifier = search.best_estimator_
        else:
            self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred
