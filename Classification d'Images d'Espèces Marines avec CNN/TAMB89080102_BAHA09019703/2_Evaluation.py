# **************************************************************************
# INF5081
# Travail pratique 2

# Par
# Tamrat Beede Mikael (TAMB89080102)
# AMADOU SARA BAH (BAHA09019703)
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

import seaborn as sns

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "/content/donnees/"
testPath = mainDataPath + "test"

# Le nombre d'images évaluées est déterminé par le contenu du dossier 'test'

# La taille des images à classer
image_scale = 150

# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,
    batch_size=32,
    color_mode=images_color_mode)

(x, y_true) = test_itr.next()

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = test_itr.classes


# evaluation du modËle
test_eval = Classifier.evaluate(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict(test_itr, verbose=1)
predicted_classes = np.argmax(predicted_classes, axis=1)  # Extraction de l'indice de la classe la plus probable


# Récupération des labels de classe pour l'utilisation dans la matrice de confusion
class_labels = list(test_itr.class_indices.keys())


# Cette list contient les images bien classées
correct = np.where(predicted_classes == y_true)[0]

# Nombre d'images bien classées
print("> %d  Ètiquettes bien classÈes" % len(correct))

# Cette list contient les images mal classées
incorrect = np.where(predicted_classes != y_true)[0]

# Nombre d'images mal classées
print("> %d Ètiquettes mal classÈes" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************

# Mise à jour pour afficher la matrice de confusion pour trois classes
cm = confusion_matrix(test_itr.classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matrice de Confusion')
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.show()

# Mise à jour pour extraire une image mal classée pour chaque combinaison d'espèces
# Afficher les images mal classées (si vous souhaitez voir les images)
incorrect_indices = np.where(predicted_classes != test_itr.classes)[0]
if incorrect_indices.size > 0:
    plt.figure(figsize=(10, 10))
    for i, index in enumerate(incorrect_indices[:9]):
        img, label = test_itr.next()  # Fetch next batch
        if label.shape[1] > 1:
            label_index = np.argmax(label[0])
        else:
            label_index = int(label[0])
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[0], interpolation='none')
        plt.title(f"Prévu: {class_labels[predicted_classes[index]]}, Vrai: {class_labels[label_index]}")
        plt.tight_layout()
    plt.show()
