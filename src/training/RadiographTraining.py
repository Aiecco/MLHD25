import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from typing import Optional, Tuple  # Import for type hinting


def train_model(model: tf.keras.Model,
                train_dataset: tf.data.Dataset,
                epochs: int = 50,
                validation_dataset: Optional[tf.data.Dataset] = None,
                model_save_path: str = 'best_age_prediction_model.keras') -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains the provided Keras model using specified datasets and callbacks.

    This function orchestrates the training process, including saving the best model,
    implementing early stopping to prevent overfitting, and dynamically adjusting
    the learning rate during training based on validation performance.

    Args:
        model (tf.keras.Model): The already compiled Keras model to be trained.
        train_dataset (tf.data.Dataset): The TensorFlow Dataset for training.
                                         It should yield (input_data, labels) tuples,
                                         where input_data matches the model's input
                                         (e.g., preprocessed images).
        epochs (int, optional): The maximum number of training epochs. Training
                                may stop earlier due to EarlyStopping. Defaults to 50.
        validation_dataset (tf.data.Dataset, optional): The TensorFlow Dataset for validation.
                                                        Used to monitor performance and guide
                                                        EarlyStopping and ReduceLROnPlateau.
                                                        Should have the same structure as `train_dataset`.
                                                        Defaults to None.
        model_save_path (str, optional): The file path where the best performing model
                                         (based on validation MAE) will be saved.
                                         Defaults to 'best_age_prediction_model.keras'.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing:
            - tf.keras.Model: The trained model instance. Its weights will be restored
                              to the best observed performance during training if EarlyStopping
                              with `restore_best_weights=True` is used.
            - tf.keras.callbacks.History: An object containing the history of loss and
                                           metric values during training.
    """
    print(f"\nStarting model training for {epochs} epochs...")

    # Define Keras Callbacks for enhanced training control and regularization.
    callbacks = [
        # ModelCheckpoint: Saves the best model weights observed during training.
        # It monitors 'val_mae' (Mean Absolute Error on validation set) and saves
        # only if the monitored metric improves (mode='min' for MAE).
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_mae' if validation_dataset else 'mae', # Monitor validation MAE if validation set exists, else training MAE
            save_best_only=True, # Only save the model when validation MAE improves
            mode='min', # 'min' mode means lower is better for the monitored metric (MAE)
            verbose=1 # Display messages when a better model is saved
        ),
        # EarlyStopping: Halts training if the monitored metric does not improve
        # for a specified number of epochs (patience). This prevents overfitting.
        EarlyStopping(
            monitor='val_mae' if validation_dataset else 'mae', # Monitor validation MAE
            patience=10, # Number of epochs with no improvement after which training will be stopped
            mode='min', # 'min' mode for MAE
            verbose=1, # Display messages when early stopping is triggered
            restore_best_weights=True # Restore model weights from the epoch with the best monitored value
        ),
        # ReduceLROnPlateau: Dynamically reduces the learning rate when a metric
        # has stopped improving. This helps the model to converge more finely.
        ReduceLROnPlateau(
            monitor='val_mae' if validation_dataset else 'mae', # Monitor validation MAE
            factor=0.5, # Factor by which the learning rate will be reduced (new_lr = lr * factor)
            patience=5, # Number of epochs with no improvement after which the learning rate will be reduced
            min_lr=1e-6, # Lower bound on the learning rate
            mode='min', # 'min' mode for MAE
            verbose=1 # Display messages when learning rate is reduced
        )
    ]

    # Start the model training process.
    # The 'initial_epoch=5' parameter means training will start from epoch 5.
    # This might be useful if resuming training and wanting to skip initial already-converged epochs
    # or to allow callbacks to become active after a few epochs.
    # If resuming from a loaded model, Keras automatically handles the initial epoch correctly.
    # It's important to ensure this parameter aligns with the overall training strategy.
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        initial_epoch=5 # This parameter means training will resume counting epochs from 5.
                        # If you are truly restarting a new training, this should be 0.
                        # If you are loading a pre-trained model and want to continue,
                        # this might be set based on the last trained epoch.
    )
    print("\nModel training completed.")
    return model, history