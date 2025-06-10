from src.Models.AgePredictionModel import AgePredictionModel
from src.dataset.RadiographDataset import RadiographDatasetBuilder
import tensorflow as tf

from src.training.RadiographTraining import train_model


def training_pipeline(base_dir_train, label_train, label_val, base_dir_val, img_sizes=256):
    builder_train = RadiographDatasetBuilder(
        base_dir=base_dir_train,
        label_csv=label_train,
        img_size=(img_sizes, img_sizes),
        batch_size=16
    )
    train_dataset = builder_train.build(train=True)

    # Builder per il Validation Set
    builder_val = RadiographDatasetBuilder(
        base_dir=base_dir_val,
        label_csv=label_val,
        img_size=(img_sizes, img_sizes),
        batch_size=16
    )
    val_dataset = builder_val.build(train=False)  # Non shufflare il validation set

    print(f"Train dataset size (batches): {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size (batches): {tf.data.experimental.cardinality(val_dataset).numpy()}")

    model_builder = AgePredictionModel(img_size=(img_sizes, img_sizes))
    model_builder.compile_model() # Compila il modello

    print("\nModel Summary:")
    model_builder.model.summary()

    # --- RICHIESTA DELLA FUNZIONE train_model SEPARATA ---
    # Richiamo la funzione train_model, passando il modello compilato
    # e i dataset preparati.

    print("\nAvvio dell'addestramento:")
    try:
        # Assicurati che la funzione train_model sia accessibile qui (es. nello stesso file)
        trained_model, history = train_model(
            model_builder.model,          # Passa l'oggetto modello compilato
            train_dataset,                # Dataset di training
            validation_dataset=val_dataset, # Dataset di validazione
            epochs=50,
            model_save_path='best_age_prediction_model_standalone.keras'
        )
        print(f"Training history keys: {history.history.keys()}")
        print(f"Il modello è stato addestrato e salvato in 'best_age_prediction_model_standalone.keras'.")
    except Exception as e:
        print(f"Errore nell'addestrare il modello: {e}")

    print("\nSetup del modello e del dataset completato.")