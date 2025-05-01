import os.path
from src.Models.OrdinalRegressor import AgeEstimator
from src.dataset.RadiographDataset import RadiographDatasetBuilder
import tensorflow as tf

from src.loss.CoralLoss import coral_ordinal_loss
from src.loss.YearLoss import years_exact_acc, years_within_one_acc, years_within_two_acc, months_mae
from src.radiomics.RadiomicPreprocess import preprocess_radiomics
from src.testing.AgePrediction import predict_and_evaluate
from src.testing.Evaluation import evaluate_age_predictions
from src.testing.RadiographTesting import test_model
from src.training.RadiographTraining import train_model
from src.training.RadiographTraining_2 import train_model_with_monitoring
from src.utils.DisplayResults import display_evaluation_results
from src.utils.LoadModel import load_saved_model
from src.utils.SaveModel import save_model_properly


def radiograph_pipeline(preprocess=False, training=False, epochs=30, batch_size=64):
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
        batch_size=64
    ).build(shuffle=True)

    val_ds = RadiographDatasetBuilder(
        base_dir="data/Val",
        label_csv="val_labels.csv",
        img_subfolder="validation_samples",
        img_size=(128, 128),
        batch_size=64
    ).build(shuffle=False)

    test_ds = RadiographDatasetBuilder(
        base_dir="data/Test",
        label_csv="test_labels.csv",
        img_subfolder="test_samples",
        img_size=(128, 128),
        batch_size=64
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
            "ordinal_output": coral_ordinal_loss,  # Loss per ord_logits (primo output)
            "month_output": 'mae',  # Loss per month_out (secondo output)
            "gender_out": 'binary_crossentropy'  # Loss per gender_pred (terzo output)
        },
        loss_weights={
            "ordinal_output": 1.0,  # Peso per la loss degli anni
            "month_output": weight_months,  # Peso per la loss dei mesi
            "gender_out": lambda_adversarial  # Peso per la loss del genere (controlla GRL)
        },
        metrics={
            'coarse_fine_head': [months_mae, years_exact_acc, years_within_one_acc, years_within_two_acc],
            # Assicurati che months_mae funzioni con i target giusti
            'gender_adversarial_head': ['accuracy']  # Applica accuracy alla predizione del genere
        }
    )

    if training:
        #trained, age_metrics_callback = train_model(model_graph, train_ds, epochs, batch_size=batch_size, validation_data=val_ds)
        trained, age_metrics_callback = train_model_with_monitoring(model_graph, train_ds, batch_size=batch_size, epochs=5, validation_data=val_ds)

        # Supponiamo che model_graph sia il tuo modello già compilato e addestrato
        save_path = "out"
        os.makedirs(save_path, exist_ok=True)
        #save_model_properly(model_graph, save_path)
        trained.save("out/age_estimator.keras")

        # Alla fine del training, visualizza l'andamento delle metriche
        age_metrics_callback.plot_metrics_history()

        # Valuta il modello sul test set con visualizzazioni dettagliate
        test_results = predict_and_evaluate(model, test_ds, plot_results=True)
        print(f"MAE finale (mesi): {test_results['MAE (mesi)']:.2f}")
        print(f"Accuratezza anno esatto: {test_results['Accuratezza anno esatto (%)']:.2f}%")


    model_path = "out"
    # Tenta prima il caricamento diretto
    loaded_model = load_saved_model(model_path)

    # Se abbiamo caricato con successo il modello, valutiamolo sul test set
    if loaded_model is not None:
        # Valuta il modello
        results = predict_and_evaluate(loaded_model, test_ds, plot_results=True)

        # Ora hai tutte le metriche e le visualizzazioni per la discussione dei risultati
        display_evaluation_results(results)
    else:
        print("Impossibile caricare il modello. Verifica i percorsi e le definizioni delle classi personalizzate.")
