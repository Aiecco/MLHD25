import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from keras.src.losses import mean_squared_error


def predict_and_evaluate(model, test_ds, plot_results=True):
    # Liste per memorizzare risultati
    true_months_list = []
    pred_months_list = []
    errors_list = []

    # Valutazione su validation/test set
    for x_batch, y_batch in test_ds:
        # y_batch è (age_months, 1)
        if isinstance(y_batch, (list, tuple)):
            y_true_months = y_batch[0].numpy()  # tensore scalare (batch_size,)
        else:
            y_true_months = y_batch.numpy()

        # Predizione (output diretto: (batch_size, 1))
        #print(f"feat_rad.shape: {x_batch[0].shape}, feat_img.shape: {x_batch[1].shape}")
        predictions = model.predict(x_batch, verbose=0)
        if isinstance(predictions, list) or isinstance(predictions, tuple):
            predictions = predictions[0]

        y_pred_months = predictions.squeeze()  # Rimuove dimensione extra se presente

        # Raccogli risultati
        true_months_list.extend(y_true_months)
        pred_months_list.extend(y_pred_months)

        # Errori
        errors = np.abs(y_true_months - y_pred_months)
        errors_list.extend(errors)

    mae_months = mean_squared_error(true_months_list, pred_months_list)
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

    # Stampa risultati
    print("\n==== RISULTATI VALUTAZIONE ====")
    print(f"MSE (mesi): {mae_months:.2f}")
    print(f"MSE (anni): {mae_years:.2f}")

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
        year_bins = np.linspace(0, 20, 4)

        for i in range(len(true_years_array)):
            true_bin = min(int(true_years_array[i] // 5), 3)
            pred_bin = min(int(pred_years_array[i] // 5), 3)
            heatmap_data[true_bin, pred_bin] += 1

        # Normalizza per riga
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        heatmap_data_norm = np.divide(heatmap_data, row_sums, out=np.zeros_like(heatmap_data), where=row_sums != 0)

        sns.heatmap(heatmap_data_norm, cmap='YlGnBu',
                    xticklabels=[f'{i}-{i + 8}' for i in range(0, 20, 5)],
                    yticklabels=[f'{i}-{i + 8}' for i in range(0, 20, 5)])
        plt.xlabel('Età predetta (anni)')
        plt.ylabel('Età vera (anni)')
        plt.title('Distribuzione delle predizioni per fascia d\'età')

        plt.tight_layout()
        plt.savefig('age_prediction_analysis.png', dpi=300)
        plt.show()

    return results