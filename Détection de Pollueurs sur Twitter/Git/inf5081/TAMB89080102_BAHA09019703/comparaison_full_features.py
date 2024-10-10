import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from model_visualizer import ModelVisualizer

class ClassifierAnalysis:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.X = None  # Features
        self.y = None  # Labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        # Convertir les dates en nombres
        if 'CreatedAt' in self.data.columns:
            self.data['CreatedAt'] = pd.to_datetime(self.data['CreatedAt'])
            self.data['CreatedAt'] = (self.data['CreatedAt'] - self.data['CreatedAt'].min()) / np.timedelta64(1, 'D')

        # Convertir les colonnes de type object en valeurs numériques
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])

        self.X = self.data.drop('Label', axis=1)
        self.y = self.data['Label']

    def split_data(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)
        probas = model.predict_proba(self.X_test)[:, 1]  # Probabilités pour la classe positive

        # Calcul des métriques
        accuracy = accuracy_score(self.y_test, predictions)
        f_measure = f1_score(self.y_test, predictions, pos_label=1)  # pos_label=1 pour Content Polluters
        auc = roc_auc_score(self.y_test, probas)

        # Matrice de confusion pour calculer TP Rate et FP Rate
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (fp + tn)

        # Préparation du rapport
        report = classification_report(self.y_test, predictions)

        return accuracy, tp_rate, fp_rate, f_measure, auc, report

    def display_results(self, model):
        accuracy, tp_rate, fp_rate, f_measure, auc, report = self.evaluate_model(model)
        print(f"Accuracy: {accuracy}")
        print(f"TP Rate: {tp_rate}")
        print(f"FP Rate: {fp_rate}")
        print(f"F-measure: {f_measure}")
        print(f"AUC: {auc}")
        print("Classification Report:")
        print(report)

    def run_analysis(self):
        self.load_and_prepare_data()
        self.split_data()

        print("Analyse avec l'arbre de décision:")
        dt_model = self.train_model(DecisionTreeClassifier())
        self.display_results(dt_model)
        dt_visualizer = ModelVisualizer(dt_model, self.X_test, self.y_test)
        dt_visualizer.plot_confusion_matrix()
        dt_visualizer.plot_roc_curve()

        print("\nAnalyse avec la forêt aléatoire:")
        rf_model = self.train_model(RandomForestClassifier())
        self.display_results(rf_model)
        rf_visualizer = ModelVisualizer(rf_model, self.X_test, self.y_test)
        rf_visualizer.plot_confusion_matrix()
        rf_visualizer.plot_roc_curve()

        print("\nAnalyse avec la classification bayésienne naïve:")
        nb_model = self.train_model(GaussianNB())
        self.display_results(nb_model)
        nb_visualizer = ModelVisualizer(nb_model, self.X_test, self.y_test)
        nb_visualizer.plot_confusion_matrix()
        nb_visualizer.plot_roc_curve()

