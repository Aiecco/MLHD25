# Per utilizzare questa funzione nel callback durante il training
import tensorflow as tf
import matplotlib.pyplot as plt

from src.testing.Evaluation import evaluate_age_predictions


class AgeMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, validation_steps=None, frequency=5):
        super(AgeMetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.frequency = frequency  # Ogni quante epoche calcolare le metriche
        self.history = {
            'epoch': [],
            'MAE (mesi)': [],
            'Accuratezza anno esatto (%)': [],
            'Accuratezza entro 1 anno (%)': [],
            'Accuratezza entro 2 anni (%)': []
        }

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:  # solo a inizio
            print("DEBUG CALLBACK log keys:", logs.keys())

            
        if (epoch + 1) % self.frequency == 0:
            # Valutazione delle metriche
            results = evaluate_age_predictions(self.model, self.validation_data, plot_results=False)

            # Aggiorna la history
            self.history['epoch'].append(epoch + 1)
            self.history['MAE (mesi)'].append(results['MAE (mesi)'])
            self.history['Accuratezza anno esatto (%)'].append(results['Accuratezza anno esatto (%)'])
            self.history['Accuratezza entro 1 anno (%)'].append(results['Accuratezza entro 1 anno (%)'])
            self.history['Accuratezza entro 2 anni (%)'].append(results['Accuratezza entro 2 anni (%)'])

            # Stampa i risultati
            print(f"\nEpoch {epoch + 1} - Metriche età:")
            print(f"MAE (mesi): {results['MAE (mesi)']:.2f}")
            print(f"MAE (anni): {results['MAE (anni)']:.2f}")
            print(f"Accuratezza anno esatto: {results['Accuratezza anno esatto (%)']:.2f}%")
            print(f"Accuratezza entro 1 anno: {results['Accuratezza entro 1 anno (%)']:.2f}%")
            print(f"Accuratezza entro 2 anni: {results['Accuratezza entro 2 anni (%)']:.2f}%")

    def plot_metrics_history(self):
        """Visualizza l'andamento delle metriche durante il training"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.history['epoch'], self.history['MAE (mesi)'])
        plt.xlabel('Epoca')
        plt.ylabel('MAE (mesi)')
        plt.title('Errore Medio Assoluto durante il training')

        plt.subplot(2, 1, 2)
        plt.plot(self.history['epoch'], self.history['Accuratezza anno esatto (%)'], label='Anno esatto')
        plt.plot(self.history['epoch'], self.history['Accuratezza entro 1 anno (%)'], label='Entro 1 anno')
        plt.plot(self.history['epoch'], self.history['Accuratezza entro 2 anni (%)'], label='Entro 2 anni')
        plt.xlabel('Epoca')
        plt.ylabel('Accuratezza (%)')
        plt.title('Accuratezza delle predizioni durante il training')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics_history.png', dpi=300)
        plt.show()
