import os.path

from src.Models.OrdinalRegressor import AgeEstimator
from src.dataset.RadiographDataset import RadiographDatasetBuilder
import tensorflow as tf

from src.loss.CoralLoss import coral_loss
from src.loss.MonthsMAE import months_mae
from src.radiomics.RadiomicPreprocess import preprocess_radiomics
from src.training.RadiographTraining import train_model


def radiograph_pipeline(preprocess=False, epochs=30):
    # Estrazione radiomiche
    if preprocess:
        preprocess_radiomics("data/Train/train_samples", "data/Train/radiomics")
        preprocess_radiomics("data/Test/test_samples", "data/Test/radiomics")
        preprocess_radiomics("data/Val/validation_samples", "data/Val/radiomics")

    # Costruisci i dataset
    train_ds = RadiographDatasetBuilder(
        base_dir="data/Train",
        label_csv="train_labels.csv",
        img_subfolder="train_samples",
        img_size=(128, 128),
        batch_size=16
    ).build(shuffle=True)

    val_ds = RadiographDatasetBuilder(
        base_dir="data/Val",
        label_csv="val_labels.csv",
        img_subfolder="validation_samples",
        img_size=(128, 128),
        batch_size=16
    ).build(shuffle=False)

    test_ds = RadiographDatasetBuilder(
        base_dir="data/Test",
        label_csv="test_labels.csv",
        img_subfolder="test_samples",
        img_size=(128, 128),
        batch_size=16
    ).build(shuffle=False)

    radiomics_dim = 4
    max_years = 100
    model = AgeEstimator(input_shape=(128, 128, 1),
                         radiomics_dim=radiomics_dim,
                         max_years=max_years)

    # Allena il modello
    model_graph = model.build_graph()
    tf.keras.utils.plot_model(model_graph, show_shapes=True)

    # --- Compilazione del Modello ---
    lambda_adversarial = - 0.1  # Peso per la loss avversaria (robustezza genere)
    weight_months = 0.5  # Peso per la loss dei mesi residui (relativo agli anni)

    model_graph.compile(
        optimizer='adam',
        loss={
            "ordinal_output": coral_loss,  # Loss per ord_logits (primo output)
            "month_output": 'mse',  # Loss per month_out (secondo output)
            "gender_out": 'binary_crossentropy'  # Loss per gender_pred (terzo output)
        },
        loss_weights={
            "ordinal_output": 1.0,  # Peso per la loss degli anni
            "month_output": weight_months,  # Peso per la loss dei mesi
            "gender_out": lambda_adversarial  # Peso per la loss del genere (controlla GRL)
        },
        metrics={
            'coarse_fine_head': [months_mae],  # Applica months_mae all'output della testa età/mesi
            # Assicurati che months_mae funzioni con i target giusti
            'gender_adversarial_head': ['accuracy']  # Applica accuracy alla predizione del genere
        }
    )

    trained = train_model(model_graph, train_ds, epochs, batch_size=16, validation_data=val_ds)

    # salvi l'intero modello (architettura + pesi + optimizer state)
    trained.save("out/age_estimator.keras")
