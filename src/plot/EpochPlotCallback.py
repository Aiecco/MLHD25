import matplotlib.pyplot as plt
import os
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

        self.train_losses.append(logs.get("loss"))
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss)

        # Clear the previous plot
        plt.clf()

        # Plot
        plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over epochs")
        plt.legend()

        # Save the figure
        filepath = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}.png")
        plt.savefig(filepath)
        print(f"Saved plot to {filepath}")
