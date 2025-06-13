# src/pipelines/training_pipeline.py

import tensorflow as tf
from typing import Optional, Tuple

# Assuming these imports are correctly resolved in your project structure
from src.Models.AgePredictionModel import AgePredictionModel  # Model architecture definition
from src.dataset.RadiographDataset import RadiographDatasetBuilder  # Dataset loading and preprocessing
from src.plot.PlotHistory import plot_training_history  # Utility for plotting training history
from src.training.RadiographTraining import train_model  # Core model training function

def training_pipeline(base_dir_train: str, label_train: str,
                      base_dir_val: str, label_val: str,
                      img_sizes: int = 256,
                      mean_pixel_value: float = 0.0, std_pixel_value: float = 1.0,
                      existing_model: Optional[tf.keras.Model] = None,
                      epochs: int = 50,
                      model_save_path: str = 'best_age_prediction_model_standalone.keras',
                      learning_rate: float = 0.0005) -> Optional[tf.keras.Model]:
    """
    Orchestrates the entire model training process.

    This pipeline handles the creation and preparation of training and validation datasets,
    the initialization or loading of the deep learning model, and the execution of the
    training loop with specified parameters and callbacks. It also plots the training history.

    Args:
        base_dir_train (str): The base directory for the training dataset, containing images and labels.
        label_train (str): The filename of the CSV file containing labels for the training set,
                           relative to `base_dir_train`.
        base_dir_val (str): The base directory for the validation dataset, containing images and labels.
        label_val (str): The filename of the CSV file containing labels for the validation set,
                         relative to `base_dir_val`.
        img_sizes (int, optional): The target side length for input images (e.g., 256 for 256x256).
                                   Defaults to 256.
        mean_pixel_value (float, optional): The mean pixel value calculated from the training set.
                                            Used for standardizing both training and validation images.
                                            Defaults to 0.0.
        std_pixel_value (float, optional): The standard deviation of pixel values from the training set.
                                           Used for standardizing both training and validation images.
                                           Defaults to 1.0.
        existing_model (tf.keras.Model, optional): An already initialized or loaded Keras model.
                                                   If provided, training continues from this model's state.
                                                   If None, a new `AgePredictionModel` is created.
                                                   Defaults to None.
        epochs (int, optional): The maximum number of epochs for which to train the model.
                                Actual epochs may be fewer due to early stopping. Defaults to 50.
        model_save_path (str, optional): The file path where the best performing model
                                         (based on validation metrics) will be saved during training.
                                         Defaults to 'best_age_prediction_model_standalone.keras'.
        learning_rate (float, optional): The initial learning rate for the optimizer if a new model
                                         is created and compiled. Defaults to 0.0005.

    Returns:
        Optional[tf.keras.Model]: The trained Keras model instance. Returns None if an error
                                  occurs during the training process.
    """

    # --- Dataset Preparation ---
    # Initialize RadiographDatasetBuilder for the training set.
    # It will load image paths and labels, and apply specified preprocessing (including standardization).
    builder_train = RadiographDatasetBuilder(
        base_dir=base_dir_train,
        label_csv=label_train,  # Expects label_csv relative to base_dir
        img_size=(img_sizes, img_sizes),
        batch_size=16,
        mean_pixel_value=mean_pixel_value,
        std_pixel_value=std_pixel_value
    )
    # Build the TensorFlow Dataset for training. `train=True` implies shuffling.
    train_dataset = builder_train.build(shuffle=True)

    # Initialize RadiographDatasetBuilder for the validation set.
    # It uses the same standardization parameters as the training set for consistency.
    builder_val = RadiographDatasetBuilder(
        base_dir=base_dir_val,
        label_csv=label_val,  # Expects label_csv relative to base_dir
        img_size=(img_sizes, img_sizes),
        batch_size=16,
        mean_pixel_value=mean_pixel_value,
        std_pixel_value=std_pixel_value
    )
    # Build the TensorFlow Dataset for validation. `train=False` implies no shuffling.
    val_dataset = builder_val.build(shuffle=False)

    # Print the calculated dataset sizes in terms of batches.
    print(f"\nTrain dataset size (batches): {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size (batches): {tf.data.experimental.cardinality(val_dataset).numpy()}")

    # --- Model Initialization or Loading ---
    model_to_train: tf.keras.Model = None
    if existing_model is not None:
        # If an existing model is provided, use it directly.
        # This allows resuming training or fine-tuning.
        model_to_train = existing_model
        print("\nUsing existing model for training.")
        # Note: If you want to recompile the existing model with new learning_rate or metrics,
        # you would do it here (e.g., model_to_train.compile(...)).
        # Keras's load_model usually preserves compilation state, but recompiling can override.
    else:
        # If no existing model is provided, create a new AgePredictionModel instance.
        model_builder = AgePredictionModel(img_size=(img_sizes, img_sizes))
        # Compile the newly created model with the specified learning rate.
        model_builder.compile_model(learning_rate=learning_rate)
        model_to_train = model_builder.model
        print("\nCreating and compiling a new model for training.")

    # Display the summary of the model (whether new or existing).
    print("\nModel Summary:")
    model_to_train.summary()

    # --- Model Training Execution ---
    print("\nStarting training:")
    try:
        # Call the core `train_model` function to execute the training loop.
        # This function handles epochs, validation, and callbacks (e.g., ModelCheckpoint, EarlyStopping).
        trained_model, history = train_model(
            model=model_to_train,  # The model to be trained
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            epochs=epochs,
            model_save_path=model_save_path
        )
        # Plot the training and validation loss/metrics history.
        plot_training_history(history, save_path='training_history_plots.png')

        print(f"Training history keys: {history.history.keys()}")
        print(f"Model trained and saved to '{model_save_path}'.")
        return trained_model  # Return the trained model instance
    except Exception as e:
        # Catch any exceptions during training and print a detailed traceback.
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return None  # Indicate that training failed by returning None