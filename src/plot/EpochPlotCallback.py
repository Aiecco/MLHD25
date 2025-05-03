import os
import matplotlib.pyplot as plt
import tensorflow as tf

class EpochPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir="img/loss_plots"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 1) Nuova figura
        plt.figure()

        # 2) Recupera e salva i valori
        train_loss = logs.get("loss")
        if train_loss is not None:
            self.train_losses.append(train_loss)

        val_loss = logs.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss)

        # 3) Disegna
        plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over epochs")
        plt.legend()

        # 4) Salva e chiudi
        filepath = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(filepath)
        plt.close()
        print(f" - Saved plot to {filepath}")
