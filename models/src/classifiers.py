from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import os
import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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
        return None    
        
    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred

class CNN:
    def __init__(self, name, input_shape = (128, 1293, 1), n_classes=10):
        self.name = name
        self.classifier = self.create_model(input_shape, n_classes)
    
    def create_model(self, input_shape, n_classes):
        
        def audio_augmentations(x: tf.Tensor) -> tf.Tensor:
            val = random.random()
            if val>0.5:
                x = tfio.audio.freq_mask(x, 10)
            else:
                x = tfio.audio.time_mask(x, 10)
            return x
        
        inputs = layers.Input(shape=input_shape)
        x = layers.Resizing(32, 32)(inputs)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(n_classes)(x)
        model = tf.keras.Model(inputs, x)
        
        return model
    
    def fit(self, X, y=None, **kwargs):
        
        train = X
        
        # compile with optimizer, loss, metrics
        self.classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'],)
        
        if len(kwargs['save_location'])>0:
            folder = f"{kwargs['save_location']}/{self.name}"
            os.makedirs(folder, exist_ok=True)
        
        # callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=folder+'-{epoch:04d}.ckpt', 
                                                         verbose=1, 
                                                         save_weights_only=True,
                                                         save_freq='epoch')
        es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
        
        # fit
        history = self.classifier.fit(X, epochs = kwargs['EPOCHS'],
                                      steps_per_epoch=kwargs['STEPS_PER_EPOCH'], 
                                      callbacks = [cp_callback, es_callback], validation_data = kwargs['val'] if 'val' in kwargs else None)
        return history
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def load(self,location):
        pass