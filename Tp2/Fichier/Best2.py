import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import time

# Configuration du GPU pour TensorFlow
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Chemins des données
mainDataPath = "/content/donnees/"
trainPath = mainDataPath + "entrainement/"
# validationPath = mainDataPath + "validation/"  # Assurez-vous que ce dossier existe et qu'il contient des données, sinon vous devrez le créer lors de la préparation des données.
testPath = mainDataPath + "test/"
modelsPath = "/content/Model.hdf5"  # Modifiez cela pour pointer vers le chemin où vous souhaitez enregistrer le modèle.

# Paramètres des images
image_scale = 150
image_channels = 3
images_color_mode = "rgb"
image_shape = (image_scale, image_scale, image_channels)

# Paramètres d'entraînement
training_batch_size = 9600
validation_batch_size = 2400
fit_batch_size = 32
fit_epochs = 40
validation_split = 0.2  # 20% des données pour la validation

# Création du modèle CNN
input_layer = Input(shape=image_shape)

def feature_extraction(input):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Adding a third convolutional block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

     # Adding a fourth convolutional block
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same')(x)  # Another new layer
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    return x

def fully_connected(encoded):
    x = Flatten()(encoded)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)
    return x

model = Model(inputs=input_layer, outputs=fully_connected(feature_extraction(input_layer)))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Préparation des générateurs de données
# Mise à jour des générateurs de données pour inclure la validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=24,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split  # Ajout de la validation split
)

# val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(image_scale, image_scale),
    batch_size=fit_batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # Spécifiez que c'est l'ensemble d'entraînement
)

# Générateur pour les données de validation
validation_generator = train_datagen.flow_from_directory(
    trainPath,  # Notez que nous utilisons toujours trainPath ici
    target_size=(image_scale, image_scale),
    batch_size=fit_batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation'  # Spécifiez que c'est l'ensemble de validation
)

# Setup EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle
start_time = time.time()
modelcheckpoint = ModelCheckpoint(filepath=modelsPath, monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
    train_generator,
    steps_per_epoch=training_batch_size // fit_batch_size,
    epochs=fit_epochs,
    validation_data=validation_generator,
    validation_steps=validation_batch_size // fit_batch_size,
    callbacks=[modelcheckpoint]  # Add early stopping here
)

end_time = time.time()

# Affichage des résultats
print(f"Temps d'entraînement total: {(end_time - start_time):.2f} secondes.")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision Validation')
plt.title('Précision du modèle par époque')
plt.ylabel('Précision')
plt.xlabel('Époque')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entraînement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.title('Perte du modèle par époque')
plt.ylabel('Perte')
plt.xlabel('Époque')
plt.legend()
plt.tight_layout()
plt.show()
 