# ================================================================
# INF5081 
# Travail pratique 2 
# ================================================================

# ================================================================
# =============CHARGEMENT DES LIBRAIRIES=========================
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf

# ================================================================
# ===================CONFIGURATION DU GPU========================
# ================================================================

config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ================================================================
# ==================CHARGEMENT DU MODÈLE=========================
# ================================================================

model_path = "/content/Model.hdf5"  # Make sure to adjust this path if necessary
model = load_model(model_path)

# ================================================================
# ==================VARIABLES DE CHEMIN===========================
# ================================================================

mainDataPath = "/content/donnees/"
testPath = mainDataPath + "test/"

# ================================================================
# =================CHARGEMENT DES IMAGES=========================
# ================================================================

test_data_generator = ImageDataGenerator(rescale=1./255)

test_generator = test_data_generator.flow_from_directory(
    testPath,
    target_size=(150, 150),  # Should match the input size of the network
    batch_size=32,
    class_mode='categorical',  # As we have more than two classes now
    shuffle=False,
    color_mode="rgb"  # Images are in RGB
)

# ================================================================
# ==================ÉVALUATION DU MODÈLE=========================
# ================================================================

# Obtaining predictions and true labels
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ================================================================
# ==================MATRICE DE CONFUSION=========================
# ================================================================

# Generating and plotting the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ================================================================
# ============EXTRACTION DES IMAGES MAL CLASSÉES==================
# ================================================================

incorrect_indices = np.where(predicted_classes != true_classes)[0]

# Plotting misclassified images for each species combination
if incorrect_indices.size > 0:
    plt.figure(figsize=(15, 10))
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_generator[0][0][incorrect], cmap='gray', interpolation='none')
        plt.title(f"Predicted: {class_labels[predicted_classes[incorrect]]}, True: {class_labels[true_classes[incorrect]]}")
        plt.tight_layout()

# ================================================================
# =====================AFFICHAGE DES RÉSULTATS===================
# ================================================================

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print('>Test Loss:', test_loss)
print('>Test Accuracy:', test_accuracy)
