from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input, Conv2D 
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

from src.utils import get_augmentation_layer


def cnn_model():
  """
  Crea un modello CNN che riceve in input il layer di data augmentation definito in utils
  ed esegue la classificazione binaria dell'immagine
  """

  clear_session()   # Pulisco la memoria ad ogni esecuzione
  cnn = Sequential([
      InputLayer(shape = (32, 32, 3)),

      get_augmentation_layer(),

      # Blocco 1: 32 filtri

      Conv2D(32, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),   # Batchnormalization stabilizza i pesi e accelera il training
      Dropout(.3),            # Dropout "spegne" il casuale 30% dei neuroni 
      Conv2D(32, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),
      Dropout(.3),
      MaxPool2D(),            # Maxpool riduce le dimensioni e spinge la rete a focalizzarsi

      # Blocco 2: 64 filtri

      Conv2D(64, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),
      Dropout(.3),
      Conv2D(64, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),
      Dropout(.3),
      MaxPool2D(),

      # Blocco 3: 128 filtri

      Conv2D(128, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),
      Dropout(.3),
      Conv2D(128, (3,3), padding = "same", activation = "relu"),
      BatchNormalization(),
      Dropout(.3),
      MaxPool2D(),

      # Classifier:

      GlobalAveragePooling2D(),
      Dropout(.2),
      Dense(512, activation = "relu"),
      Dropout(.3),
      Dense(1, activation = "sigmoid")
  ])
  return cnn