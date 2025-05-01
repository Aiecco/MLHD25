import h5py
import os

def print_hdf5_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print("  -> Dataset Shape:", obj.shape, "Dtype:", obj.dtype)
    # Stampa anche gli attributi se presenti (potrebbero contenere nomi dei pesi)
        for key, val in obj.attrs.items():
            print(f"    -> Attr: {key} = {val}")

def load_structure():
    model_path = "out"
    weights_file = os.path.join(model_path, "age_estimator.weights.h5")  # o .h5
    if os.path.exists(weights_file):
        print(f"--- Ispezionando {weights_file} ---")
        with h5py.File(weights_file, 'r') as f:
            # Stampa la struttura gerarchica
            f.visititems(print_hdf5_structure)
            print("\n--- Fine Ispezione ---")

            # Cerca specificamente i pesi del layer problematico
            # Il nome esatto potrebbe variare (es. age_estimator/coarse_fine_head/...)
            layer_name_in_h5 = "coarse_fine_head" # Potrebbe essere annidato, es. "age_estimator/coarse_fine_head"
            print(f"\n--- Cercando '{layer_name_in_h5}' ---")
            if layer_name_in_h5 in f:
                 head_group = f[layer_name_in_h5]
                 print(f"Trovato gruppo: {layer_name_in_h5}")
                 if 'ordinal_years' in head_group:
                     ordinal_group = head_group['ordinal_years']
                     print("  Trovato sotto-gruppo 'ordinal_years'")
                     if 'kernel:0' in ordinal_group and 'bias:0' in ordinal_group:
                         print("    -> Trovati kernel:0 e bias:0!")
                     else:
                         print("    -> ERRORE: kernel:0 o bias:0 MANCANTI in 'ordinal_years'!")
                         print(f"    Contenuto di 'ordinal_years': {list(ordinal_group.keys())}")

                 else:
                     print(f"  ERRORE: 'ordinal_years' non trovato dentro '{layer_name_in_h5}'")
                     print(f"  Contenuto di '{layer_name_in_h5}': {list(head_group.keys())}")

            else:
                 print(f"ERRORE: Gruppo '{layer_name_in_h5}' non trovato nel file H5.")
                 print(f"Gruppi principali nel file: {list(f.keys())}")

    else:
        print(f"ERRORE: File dei pesi non trovato: {weights_file}")