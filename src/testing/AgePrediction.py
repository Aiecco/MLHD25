import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def predict_and_evaluate(model, test_ds, plot_results=True):
    """
    Esegue predizioni sul test set e valuta le performance
    """
    # Liste per memorizzare risultati
    true_months_list = []
    pred_months_list = []
    errors_list = []
    true_years_list = []
    pred_years_list = []

    # Estrazione delle predizioni
    for batch in test_ds:
        # Ottieni input e output
        if isinstance(batch, tuple):
            if len(batch) == 2:  # (x, y)
                x, y = batch
            else:  # (x, y, _) o altro
                x, y = batch[0], batch[1]
        else:
            raise ValueError("Formato del dataset non riconosciuto")

        # Gestisci il formato di input
        if isinstance(x, dict):
            inputs = x
        else:
            # Se è un elenco di tensori, assumiamo che siano [img, rad]
            inputs = {'radiograph': x[0], 'radiomics': x[1]}

        # Gestisci il formato del target
        if isinstance(y, dict):
            y_true_months = y.get('coarse_fine_head', y.get('total_months', None))
            y_true_years = y.get('ordinal_output', None)
        elif isinstance(y, (list, tuple)):
            y_true_months = y[0]
            y_true_years = None
        else:
            y_true_months = y
            y_true_years = None

        # Fai la predizione
        inputs_batched = {
            'radiograph': tf.expand_dims(inputs['radiograph'], 0),
            'radiomics': tf.expand_dims(inputs['radiomics'], 0)
        }
        predictions = model.predict(inputs_batched, verbose=0)

        # Estrai le predizioni rilevanti
        if isinstance(predictions, dict):
            ord_logits = predictions.get('ordinal_output', None)
            month_pred = predictions.get('month_output', None)
        elif isinstance(predictions, (list, tuple)):
            ord_logits, month_pred = predictions[0], predictions[1]
        else:
            raise ValueError("Formato di output del modello non supportato")

        # Calcola i mesi totali stimati
        est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
        est_months = est_years * 12.0 + tf.squeeze(month_pred, -1)

        # Estrai gli anni stimati e veri per confronti
        if y_true_years is not None:
            true_years = tf.reduce_sum(tf.cast(y_true_years >= 1, tf.float32), axis=1)
        else:
            true_years = tf.floor(y_true_months / 12.0)

        # Calcola gli errori
        errors = tf.abs(est_months - y_true_months)

        # Aggiungi alle liste
        true_months_list.extend(y_true_months.numpy().flatten())
        pred_months_list.extend(est_months.numpy().flatten())
        errors_list.extend(errors.numpy().flatten())
        true_years_list.extend(true_years.numpy().flatten())
        pred_years_list.extend(est_years.numpy().flatten())

    # Calcola metriche
    from sklearn.metrics import mean_absolute_error
    mae_months = mean_absolute_error(true_months_list, pred_months_list)
    mae_years = mae_months / 12.0

    # Calcola percentuali di accuratezza per diverse soglie
    accuracy_exact_year = np.mean(np.array(true_years_list) == np.array(pred_years_list)) * 100
    within_1_year = np.mean(np.abs(np.array(true_years_list) - np.array(pred_years_list)) <= 1) * 100
    within_2_years = np.mean(np.abs(np.array(true_years_list) - np.array(pred_years_list)) <= 2) * 100

    results = {
        'MAE (mesi)': mae_months,
        'MAE (anni)': mae_years,
        'Accuratezza anno esatto (%)': accuracy_exact_year,
        'Accuratezza entro 1 anno (%)': within_1_year,
        'Accuratezza entro 2 anni (%)': within_2_years,
        'Predizioni': {
            'età_vera_mesi': true_months_list,
            'età_pred_mesi': pred_months_list,
            'errori_assoluti': errors_list,
            'età_vera_anni': true_years_list,
            'età_pred_anni': pred_years_list
        }
    }

    # Stampa risultati
    print("\n==== RISULTATI VALUTAZIONE ====")
    print(f"MAE (mesi): {mae_months:.2f}")
    print(f"MAE (anni): {mae_years:.2f}")
    print(f"Accuratezza anno esatto: {accuracy_exact_year:.2f}%")
    print(f"Accuratezza entro ±1 anno: {within_1_year:.2f}%")
    print(f"Accuratezza entro ±2 anni: {within_2_years:.2f}%")

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
        age_bins = [0, 36, 72, 120, 180, 240, 300, 360, 420, 480, 600, 720, 960]
        age_labels = ['0-3', '3-6', '6-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-50', '50-60',
                      '60+']

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
        heatmap_data = np.zeros((10, 10))
        year_bins = np.linspace(0, 80, 11)

        for i in range(len(true_years_array)):
            true_bin = min(int(true_years_array[i] // 8), 9)
            pred_bin = min(int(pred_years_array[i] // 8), 9)
            heatmap_data[true_bin, pred_bin] += 1

        # Normalizza per riga
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        heatmap_data_norm = np.divide(heatmap_data, row_sums, out=np.zeros_like(heatmap_data), where=row_sums != 0)

        sns.heatmap(heatmap_data_norm, cmap='YlGnBu',
                    xticklabels=[f'{i}-{i + 8}' for i in range(0, 80, 8)],
                    yticklabels=[f'{i}-{i + 8}' for i in range(0, 80, 8)])
        plt.xlabel('Età predetta (anni)')
        plt.ylabel('Età vera (anni)')
        plt.title('Distribuzione delle predizioni per fascia d\'età')

        plt.tight_layout()
        plt.savefig('age_prediction_analysis.png', dpi=300)
        plt.show()

    return results