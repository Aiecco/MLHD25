# Funzione per valutare il modello e raccogliere le predizioni e gli errori
from src.utils.TotalMonths import calculate_total_months
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error


def evaluate_age_predictions(model, val_dataset, plot_results=True):
    # Liste per memorizzare risultati
    true_months_list = []
    pred_months_list = []
    errors_list = []

    # Valutazione su validation/test set
    for x_batch, y_batch in val_dataset:
        # Estrai le età vere in mesi (suppongo che sia il primo elemento di y_batch)
        if isinstance(y_batch, dict):
            # Se stai usando un dizionario di output
            y_true_months = y_batch.get('total_months', None)
            if y_true_months is None and 'coarse_fine_head' in y_batch:
                y_true_months = y_batch['coarse_fine_head']
        elif isinstance(y_batch, (list, tuple)):
            # Se stai usando una lista di output
            y_true_months = y_batch[0]  # Adatta questo indice alla tua struttura
        else:
            # Se è un singolo output
            y_true_months = y_batch

        # Fai la predizione
        predictions = model.predict(x_batch, verbose=0)

        # Estrai le predizioni rilevanti
        if isinstance(predictions, dict):
            month_pred = predictions.get('month_output', predictions.get('month_pred', None))
        elif isinstance(predictions, (list, tuple)):
            month_pred = predictions[1]
        else:
            # Se il modello restituisce un singolo output strutturato
            raise ValueError("Formato di output del modello non supportato")

        # Calcola i mesi totali stimati
        est_months = month_pred

        # Calcola gli errori
        errors = tf.abs(est_months - y_true_months)

        # Aggiungi alle liste
        true_months_list.extend(y_true_months.numpy().flatten())
        pred_months_list.extend(est_months.flatten())
        errors_list.extend(errors.numpy().flatten())

    # Calcola metriche
    mae_months = mean_absolute_error(true_months_list, pred_months_list)
    mae_years = mae_months / 12.0

    results = {
        'MAE (mesi)': mae_months,
        'MAE (anni)': mae_years,
        'Predizioni': {
            'età_vera_mesi': true_months_list,
            'età_pred_mesi': pred_months_list,
            'errori_assoluti': errors_list,
        }
    }

    # Visualizzazioni
    if plot_results:
        plt.figure(figsize=(18, 12))

        # 1. Distribuzione degli errori assoluti
        plt.subplot(2, 2, 1)
        sns.histplot(errors_list, kde=True, bins=30)
        plt.axvline(mae_months, color='r', linestyle='--', label=f'MAE: {mae_months:.2f} mesi')
        plt.xlabel('Errore assoluto (mesi)')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione degli errori di predizione dell\'età')
        plt.legend()

        # 2. Scatter plot: età vera vs predetta
        plt.subplot(2, 2, 2)
        plt.scatter(true_months_list, pred_months_list, alpha=0.5)
        plt.plot([0, max(true_months_list)], [0, max(true_months_list)], 'r--')
        plt.xlabel('Età vera (mesi)')
        plt.ylabel('Età predetta (mesi)')
        plt.title('Età vera vs Età predetta')

        # 3. Box plot degli errori per fasce d'età
        plt.subplot(2, 2, 3)

        # Crea fasce d'età
        age_bins = [0, 36, 72, 120, 180, 240]
        age_labels = ['0-3', '3-6', '6-10', '10-15', '15-20']

        # Crea una serie pandas per l'età
        age_series = pd.Series(true_months_list)
        binned_ages = pd.cut(age_series, bins=age_bins, labels=age_labels)

        # Crea un DataFrame con età e errori
        error_df = pd.DataFrame({
            'Fascia_età': binned_ages,
            'Errore_assoluto': errors_list
        })

        # Box plot
        sns.boxplot(x='Fascia_età', y='Errore_assoluto', data=error_df)
        plt.xlabel('Fascia d\'età (anni)')
        plt.ylabel('Errore assoluto (mesi)')
        plt.title('Distribuzione degli errori per fascia d\'età')
        plt.xticks(rotation=45)

        # 4. Heatmap: errore in base all'età vera vs predetta
        plt.subplot(2, 2, 4)

        # Converti mesi in anni per leggibilità
        true_years_array = np.array(true_months_list) / 12
        pred_years_array = np.array(pred_months_list) / 12

        # Crea una matrice per la heatmap
        heatmap_data = np.zeros((4, 4))

        for i in range(len(true_years_array)):
            true_bin = min(int(true_years_array[i] // 3), 4)
            pred_bin = min(int(pred_years_array[i] // 3), 4)
            heatmap_data[true_bin, pred_bin] += 1

        # Normalizza per riga
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        heatmap_data_norm = np.divide(heatmap_data, row_sums, out=np.zeros_like(heatmap_data), where=row_sums != 0)

        sns.heatmap(heatmap_data_norm, cmap='YlGnBu',
                    xticklabels=[f'{i}-{i + 4}' for i in range(0, 20, 4)],
                    yticklabels=[f'{i}-{i + 4}' for i in range(0, 20, 4)])
        plt.xlabel('Età predetta (anni)')
        plt.ylabel('Età vera (anni)')
        plt.title('Distribuzione delle predizioni per fascia d\'età')

        plt.tight_layout()
        plt.savefig('age_prediction_analysis.png', dpi=300)
        plt.show()

    return results