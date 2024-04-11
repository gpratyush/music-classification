from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

class TraditionalModel:
    def __init__(self, name, classifier) -> None:
        self.name = name
        model_match = {"HistGradientBoostingClassifier": HistGradientBoostingClassifier()}
        if type(classifier) == str:
            self.classifier = model_match[classifier]
        else:
            self.classifier = classifier
        self.search = None
    
    def fit(self, X, y, n_folds = None, grid_search = {}, metric = None):
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
