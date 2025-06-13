# src/plot/plot_training_history.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.callbacks import History # For type hinting

def plot_training_history(history: History, save_path: str = 'training_history_plots.png'):
    """
    Generates and saves plots of the loss and MAE from the training and validation history.

    Args:
        history (tf.keras.callbacks.History): The History object returned by model.fit().
        save_path (str): The full path where the plot will be saved.
    """
    if not isinstance(history, History):
        print("Error: 'history' must be an instance of tf.keras.callbacks.History.")
        return

    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(15, 6))

    # Plot of Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], label='Training Loss')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'], label='Validation Loss')
    plt.title('Loss Curve During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)

    # Plot of MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['mae'], label='Training MAE')
    if 'val_mae' in hist:
        plt.plot(epochs, hist['val_mae'], label='Validation MAE')
    plt.title('MAE Curve During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (Months)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nTraining history plots saved as '{save_path}'.")
    plt.show() # Display the plot