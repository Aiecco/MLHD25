import math


def display_evaluation_results(results):
    """
    Mostra in formato leggibile i risultati di valutazione di un modello di stima dell'età.

    Args:
        results (dict): Un dizionario contenente i risultati di valutazione
                        con la seguente struttura:
                        {
                            'MAE (mesi)': float,
                            'MAE (anni)': float,
                            'Accuratezza anno esatto (%)': float,
                            'Accuratezza entro 1 anno (%)': float,
                            'Accuratezza entro 2 anni (%)': float,
                            'Predizioni': {
                                'età_vera_mesi': list,
                                'età_pred_mesi': list,
                                'errori_assoluti': list,
                                'età_vera_anni': list,
                                'età_pred_anni': list
                            }
                        }
    """
    print("\n" + "="*40)
    print("   RISULTATI VALUTAZIONE MODELLO")
    print("="*40)

    # Mostra le metriche scalari
    print("\n--- Metriche Globali ---")
    if 'MAE (mesi)' in results:
        print(f"  MAE (mesi): {results['MAE (mesi)']:.2f}")
    if 'MAE (anni)' in results:
        print(f"  MAE (anni): {results['MAE (anni)']:.2f}")
    if 'Accuratezza anno esatto (%)' in results:
        print(f"  Accuratezza anno esatto: {results['Accuratezza anno esatto (%)']:.2f}%")
    if 'Accuratezza entro 1 anno (%)' in results:
        print(f"  Accuratezza entro 1 anno: {results['Accuratezza entro 1 anno (%)']:.2f}%")
    if 'Accuratezza entro 2 anni (%)' in results:
        print(f"  Accuratezza entro 2 anni: {results['Accuratezza entro 2 anni (%)']:.2f}%")

    # Mostra un campione delle predizioni
    if 'Predizioni' in results and results['Predizioni']:
        preds = results['Predizioni']
        true_months = preds.get('età_vera_mesi', [])
        pred_months = preds.get('età_pred_mesi', [])
        errors = preds.get('errori_assoluti', [])
        true_years = preds.get('età_vera_anni', [])
        pred_years = preds.get('età_pred_anni', [])

        num_samples = len(true_months)
        samples_to_show = min(num_samples, 10) # Mostra al massimo 10 esempi

        if num_samples > 0:
            print(f"\n--- Dettagli Predizioni (Primi {samples_to_show} Esempi) ---")
            for i in range(samples_to_show):
                # Assicurati che le liste abbiano la stessa lunghezza
                if i < len(true_months) and i < len(pred_months) and i < len(errors):
                    true_m = true_months[i]
                    pred_m = pred_months[i]
                    error = errors[i] # Questo è già l'errore assoluto in mesi

                    # Calcola anni e mesi per una visualizzazione più leggibile
                    true_y = math.floor(true_m / 12)
                    true_rem_m = true_m % 12
                    pred_y = math.floor(pred_m / 12)
                    pred_rem_m = pred_m % 12

                    print(f"\n  Esempio {i + 1}:")
                    print(f"    Età Vera: {true_y} anni, {true_rem_m} mesi ({true_m} mesi totali)")
                    print(f"    Età Predetta: {pred_y} anni, {pred_rem_m} mesi ({pred_m} mesi totali)")
                    print(f"    Errore Assoluto: {error:.2f} mesi")

            if num_samples > samples_to_show:
                 print(f"\n(Mostrati {samples_to_show} esempi su un totale di {num_samples})")
        else:
            print("\n--- Dettagli Predizioni ---")
            print("Nessun dato di predizione disponibile.")


    print("\n" + "="*40 + "\n")