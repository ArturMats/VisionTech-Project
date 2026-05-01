# Riconoscimento di animali per auto a guida autonoma

## Dettagli del Progetto:

### Dataset:
Utilizzo del dataset CIFAR, contenente migliaia di immagini etichettate in varie categorie, inclusi veicoli e animali.
###Algoritmo:
Implementazione di una rete neurale convoluzionale (CNN) per l'analisi e la classificazione delle immagini.
###Output:
Il sistema classificherà correttamente ogni immagine come veicolo o animale.

## Valutazione del Modello:

Accuratezza: Proporzione di immagini classificate correttamente rispetto al totale.
Precisione: Qualità delle predizioni positive, indicando la proporzione di immagini correttamente identificate.
Analisi dei Risultati:

## Identificazione di eventuali pattern di errore.
Valutazione delle categorie di immagini confuse sistematicamente.
Esame delle immagini errate e riflessione su possibili migliorie al modello.
Risultato Finale:

Presentazione completa della rete neurale convoluzionale e delle sue capacità di discriminazione tra veicoli e animali.


Per il progetto si è scelto di creare una rete CNN a tre blocchi con numero di filtri crescente (32, 64 e 128) per 
consentire l'apprendimento di pattern nei minimi dettagli. Sono stati implementati i blocchi di BatchNormalization
per stabilizzare i pesi e incrementare l'apprendimento. L'aggiunta di Dropout che "spegneva" in maniera casuale
il 30% dei neuroni (e nel caso prima del layer di classificazione del 20%) garantiva che il modello non imparasse a memoria
ma che cercasse i pattern.

## RISULTATI E PERFORMANCE:

Accuracy: 96.45%
Precision: 99.07%
Recall: 95.57%
Loss: 0.1224

Questi numeri confermano l'alta affidabilità della rete nel ridurre falsi positivi garantendo che le segnalazioni di 
pericolo siano precise

Per motivi di spazio i pesi in formato .h5 sono stati salvati esternamente (salvati su Google Drive collegato al Notebook)

## INTERFACCIA DEL MODELLO IMPLEMENTATO

Per una verifica interattiva del modello è stato usato Gradio che permette di caricare l'immagine per verificare la classificazione.

