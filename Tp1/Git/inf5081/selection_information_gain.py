import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FeatureSelector:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = None  # Features
        self.y = None  # Labels

    def prepare_data(self):
        # Convertir les dates en nombres
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    self.data[col] = (self.data[col] - self.data[col].min()) / np.timedelta64(1, 'D')
                except ValueError:
                    encoder = LabelEncoder()
                    self.data[col] = encoder.fit_transform(self.data[col])

        # 'Label' est la colonne cible
        self.X = self.data.drop('Label', axis=1)
        self.y = self.data['Label']

    def select_features_with_info_gain(self, k=7):
        # Calcul du gain d'information
        info_gain = mutual_info_classif(self.X, self.y)

        # Tri des indices selon l'importance (gain d'information)
        indices = np.argsort(info_gain)[-k:]

        # Sélection des caractéristiques les plus importantes
        top_features = self.X.columns[indices]
        print(f"Top {k} features selected by information gain:")
        for feature in top_features:
            print(feature)

        return self.X[top_features]

