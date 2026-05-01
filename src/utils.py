import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

    # Creazione maschere di filtraggio:
    mask_train = np.isin(y_train.flatten(), target_indexes)
    mask_test = np.isin(y_test.flatten(), target_indexes)

    # Applicazione maschera:
    x_train_bin = x_train[mask_train]
    y_train_bin = y_train[mask_train]
    x_test_bin = x_test[mask_test]
    y_test_bin = y_test[mask_test] 

    # Binarizzazione: 0 = Veicolo, 1 = Animale
    y_train_bin = np.isin(y_train_bin, animal_indexes).astype(int)
    y_test_bin = np.isin(y_test_bin, animal_indexes).astype(int)

    # Normalizzazione pixel:
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

    # Selezione indici casuali:
    random_indices = np.random.choice(len(x_train_bin), 10, replace=False)

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train_bin[idx])
        
        # Recupero il valore 0 o 1:
        label_idx = int(y_train_bin[idx]) 
        
        # Uso il valore per pescare da target_names:
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

def callbacks():
    """
    Implementa l'earlystopping per non far correre il modello inutilmente,
    usa le metriche migliori ottenute in fase di addestramento e dimezza il learning
    rate se dopo 5 epoche non scende grazie al ReduceLROnPlateau
    """
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

def plot_learning_curves(hist,exp_name):
    """
    Funzione di visualizzazione grafica delle metriche di loss e accuracy in base 
    alle epoche
    """
    plt.figure(figsize=(10,4))
    for subplot,curve in enumerate(['loss','accuracy']):
        plt.subplot(1,2,subplot+1)
        plt.plot(hist.history[curve],label='training')
        plt.plot(hist.history['val_'+curve],label='validation')
        plt.legend()
        plt.title(exp_name+':'+curve)
    plt.tight_layout();
