import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast

def load_and_preprocess_data():
    """
    Carica CIFAR-10, filtra le classi per il progetto VisionTech (Animali vs Veicoli),
    effettua la binarizzazione e la normalizzazione.
    """
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Indici scelti: 
    # Animali: cat (3), deer (4), dog (5), horse (7)
    # Veicoli: automobile (1), truck (9)
    animal_indexes = [3, 4, 5, 7]
    vehicle_indexes = [1, 9]
    target_indexes = animal_indexes + vehicle_indexes

    # Creazione maschere di filtraggio
    mask_train = np.isin(y_train.flatten(), target_indexes)
    mask_test = np.isin(y_test.flatten(), target_indexes)

    # Applicazione maschera
    x_train_bin = x_train[mask_train]
    y_train_bin = y_train[mask_train]
    x_test_bin = x_test[mask_test]
    y_test_bin = y_test[mask_test] # Corretto: prima puntava a x_test

    # Binarizzazione: 0 = Veicolo, 1 = Animale
    y_train_bin = np.isin(y_train_bin, animal_indexes).astype(int)
    y_test_bin = np.isin(y_test_bin, animal_indexes).astype(int)

    # Normalizzazione pixel
    x_train_bin = x_train_bin.astype('float32') / 255.0
    x_test_bin = x_test_bin.astype('float32') / 255.0

    return (x_train_bin, y_train_bin), (x_test_bin, y_test_bin)

def get_augmentation_layer():
    """
    Ritorna un modello Sequential di Data Augmentation da integrare come 
    primo blocco della CNN.
    """
    return Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1)
    ], name="data_augmentation")


def show_associates(x_train_bin, y_train_bin):
    """
    Visualizza 10 immagini casuali dal set fornito per verificare 
    la corretta associazione delle etichette.
    """
    target_names = ['Veicolo', 'Animale']

    plt.figure(figsize=(12, 6))

    # Selezione indici casuali
    random_indices = np.random.choice(len(x_train_bin), 10, replace=False)

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train_bin[idx])
        
        # Recupero il valore 0 o 1
        label_idx = int(y_train_bin[idx]) 
        
        # Uso il valore per pescare da target_names
        plt.title(f"{target_names[label_idx]} ({label_idx})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def fix_random_seeds(seed=42):
    """
    Fissa i seed di tutte le librerie per garantire la riproducibilità dei
    risultati
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
