# src/plot/plot_evaluation.py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_eval(errors_list, mae_months, true_months_list, pred_months_list):
    """
        Genera e salva una serie di plot per analizzare le prestazioni del modello
        di predizione dell'età.

        Args:
            errors_list (list/array): Lista degli errori assoluti di predizione (mesi).
            mae_months (float): Mean Absolute Error complessivo (mesi).
            true_months_list (list/array): Lista delle età vere (mesi).
            pred_months_list (list/array): Lista delle età predette (mesi).
        """
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
    # Linea y=x ideale
    min_val = min(min(true_months_list), min(pred_months_list))
    max_val = max(max(true_months_list), max(pred_months_list))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideale')
    plt.xlabel('Età vera (mesi)')
    plt.ylabel('Età predetta (mesi)')
    plt.title('Età vera vs Età predetta')
    plt.legend()

    # 3. Box plot degli errori per fasce d'età
    plt.subplot(2, 2, 3)

    # Crea fasce d'età
    # Ho esteso le fasce per coprire un range più ampio se necessario
    age_bins = [0, 36, 72, 120, 180, 240, 300]  # Anni: 0-3, 3-6, 6-10, 10-15, 15-20, 20-25
    age_labels = ['0-3', '3-6', '6-10', '10-15', '15-20', '20+']  # Corrispondenti ai bin

    # Assicurati che age_bins copra l'intero range di età dei tuoi dati
    if max(true_months_list) > age_bins[-1]:
        # Aggiungi un bin finale se ci sono età superiori al massimo definito
        age_bins.append(int(max(true_months_list) + 12))  # Aggiungi un anno in più al max
        age_labels.append(f'{age_labels[-1]}+')  # Aggiorna la label per l'ultimo bin

    # Crea una serie pandas per l'età
    age_series = pd.Series(true_months_list)
    binned_ages = pd.cut(age_series, bins=age_bins, labels=age_labels, right=False)  # right=False per intervallo [a, b)

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

    # Definisci i bin per gli anni (es. 0-4, 4-8, 8-12, 12-16, 16-20, 20+)
    # Usa un passo per i bin e assicurati che coprano l'intero range di età
    max_age_in_years = max(np.max(true_years_array), np.max(pred_years_array))
    # Crea bin dinamici che includano tutte le età presenti
    # Esempio: bin ogni 4 anni, ma l'ultimo bin include il massimo
    bins_years = np.arange(0, max_age_in_years + 5, 4)  # Aggiungi un po' di margine
    if max_age_in_years > bins_years[-1]:  # Assicurati che l'ultimo bin copra il massimo
        bins_years = np.append(bins_years, max_age_in_years + 1)

    # Calcola le distribuzioni 2D con np.histogram2d
    heatmap, xedges, yedges = np.histogram2d(true_years_array, pred_years_array, bins=[bins_years, bins_years])

    # Normalizza per riga per mostrare la distribuzione delle predizioni per ogni vera fascia d'età
    row_sums = heatmap.sum(axis=1, keepdims=True)
    heatmap_norm = np.divide(heatmap, row_sums, out=np.zeros_like(heatmap), where=row_sums != 0)

    # Crea le etichette per gli assi della heatmap
    x_labels = [f'{int(xedges[i])}-{int(xedges[i + 1])}' if i < len(xedges) - 2 else f'{int(xedges[i])}+' for i in
                range(len(xedges) - 1)]
    y_labels = [f'{int(yedges[i])}-{int(yedges[i + 1])}' if i < len(yedges) - 2 else f'{int(yedges[i])}+' for i in
                range(len(yedges) - 1)]

    sns.heatmap(heatmap_norm, cmap='YlGnBu', annot=True, fmt=".2f",
                xticklabels=x_labels,
                yticklabels=y_labels)
    plt.xlabel('Età predetta (anni)')
    plt.ylabel('Età vera (anni)')
    plt.title('Distribuzione delle predizioni per fascia d\'età (Normalizzato per riga)')

    plt.tight_layout()
    plt.savefig('age_prediction_analysis.png', dpi=300)
    plt.show()