# src/plot/plot_training_history.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.callbacks import History # Per tipo hinting

def plot_training_history(history: History, save_path: str = 'training_history_plots.png'):
    """
    Genera e salva i plot della loss e MAE del training e validation history.

    Args:
        history (tf.keras.callbacks.History): L'oggetto History restituito da model.fit().
        save_path (str): Il percorso completo dove salvare il grafico.
    """
    if not isinstance(history, History):
        print("Errore: 'history' deve essere un'istanza di tf.keras.callbacks.History.")
        return

    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(15, 6))

    # Plot della Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], label='Loss di Training')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'], label='Loss di Validazione')
    plt.title('Curva di Loss durante il Training')
    plt.xlabel('Epoca')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)

    # Plot della MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['mae'], label='MAE di Training')
    if 'val_mae' in hist:
        plt.plot(epochs, hist['val_mae'], label='MAE di Validazione')
    plt.title('Curva di MAE durante il Training')
    plt.xlabel('Epoca')
    plt.ylabel('MAE (Mesi)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nGrafici della cronologia di training salvati come '{save_path}'.")
    plt.show() # Mostra il plot