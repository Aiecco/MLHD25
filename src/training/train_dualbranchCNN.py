import os
import tensorflow as tf
from keras import optimizers, losses, metrics, callbacks
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, LearningRateScheduler
)

from src.utils.Custom_MAE import CustomMAE
from src.utils.WarmUpCosine import WarmUpCosineDecayScheduler
from src.utils.data_augmentation import apply_augmentation
from src.utils.filelabels_search import filelabels_search
from src.utils.load_tensors import load_tensor
from src.preprocessing.preprocess import preprocess_dataset
from src.preprocessing.preprocess_images import preprocess_pooled_and_heatmap
from src.Models.DualBranchCNN import DualBranchCNN


def train_model(model, train_files_path='data/Train/tensors', val_files_path='data/Val/tensors', num_epochs=50,
                batch_size=32, learning_rate=1e-3, save_path="out/dual_branch_cnn_model.keras", 
                enable_augmentation=True, l2_reg=0.001, patience=10):
    """
    Funzione per addestrare il modello e salvare il modello addestrato (TensorFlow) con validation.
    
    Parametri:
    - model: modello DualBranchCNN da addestrare
    - train_files_path: percorso dei file di training
    - val_files_path: percorso dei file di validation
    - num_epochs: numero di epoche
    - batch_size: dimensione batch
    - learning_rate: learning rate iniziale
    - save_path: percorso dove salvare il modello
    - enable_augmentation: abilitare data augmentation
    - l2_reg: regolarizzazione L2
    - patience: numero di epoche senza miglioramento dopo le quali fermare il training
    """

    # Carica i dati di training
    train_files = [f for f in os.listdir(train_files_path) if f.endswith('.data-00000-of-00001')] #cambiato l'estensione
    train_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    # Ensure path compatibility across platforms
    labels_path = os.path.normpath(os.path.join(train_files_path, "..", "train_labels.csv"))
    heatmaps_path = os.path.normpath(os.path.join(train_files_path, "..", "heatmaps"))
    
    print(f"Loading {len(train_files)} training files from {train_files_path}")
    for file in train_files:
        file_prefix = os.path.join(train_files_path, file.split(".data")[0]) #creo il prefisso corretto
        try:
            id_img = int(file.split("_")[1].split(".data")[0]) #estraggo l'id_img
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
            
        try:
            pooled_tensor = load_tensor(file_prefix.replace("heatmaps", "tensors"), "tensor/.ATTRIBUTES/VARIABLE_VALUE")
            heatmap_tensor = load_tensor(file_prefix, "heated/.ATTRIBUTES/VARIABLE_VALUE")

            gender, age = filelabels_search(labels_path, id_img)
            
            # Ensure tensors have correct shapes
            pooled = tf.transpose(pooled_tensor, perm=[1, 2, 0])
            heatmap = tf.transpose(heatmap_tensor, perm=[1, 2, 0])
            
            # Validate tensor shapes
            if pooled.shape[0] != 128 or pooled.shape[1] != 128 or pooled.shape[2] != 1:
                print(f"Warning: Pooled tensor for training file {file} has incorrect shape {pooled.shape}, expected (128, 128, 1)")
                continue
                
            if heatmap.shape[0] != 128 or heatmap.shape[1] != 128 or heatmap.shape[2] != 1:
                print(f"Warning: Heatmap tensor for training file {file} has incorrect shape {heatmap.shape}, expected (128, 128, 1)")
                continue
                
            train_data["pooled"].append(pooled)
            train_data["heatmap"].append(heatmap)
            train_data["gender"].append(gender)
            train_data["age"].append(age)
            
        except Exception as e:
            print(f"Error processing training file {file}: {str(e)}")
            continue

    # Define gender mapping
    gender_mapping = {"M": 1, "F": 0}
    
    # Process gender values
    train_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in train_data["gender"]]
    
    # Check if we have training data
    if len(train_data["pooled"]) == 0:
        raise ValueError("No valid training data found. Check training files and paths.")
    
    print(f"Loaded {len(train_data['pooled'])} training samples successfully")
    
    # Stack tensors and convert to proper types
    train_pooled_tensors = tf.stack(train_data["pooled"])
    train_heatmap_tensors = tf.stack(train_data["heatmap"])
    train_gender_tensors = tf.convert_to_tensor(train_data["gender"], dtype=tf.float32)
    train_age_tensors = tf.convert_to_tensor(train_data["age"], dtype=tf.float32)
    
    # Reshape gender tensor for consistency
    train_gender_tensors = tf.reshape(train_gender_tensors, [-1, 1])
    # Crea dataset di training usando il formato input/output richiesto dal modello
    # Il modello richiede inputs con nomi specifici e un target
    train_inputs = {
        'pooled_input': train_pooled_tensors,
        'heatmap_input': train_heatmap_tensors,
        'gender_input': train_gender_tensors
    }
    train_outputs = train_age_tensors
    
    # Crea dataset usando la struttura corretta
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs))
    
    if enable_augmentation:
        # Applica data augmentation solo al dataset di training
        # Preserva la struttura degli input
        def augment_map_fn(inputs_dict, age_target):
            # Estrai i tensori individuali dal dizionario di input
            pooled = inputs_dict['pooled_input']
            heatmap = inputs_dict['heatmap_input']
            gender = inputs_dict['gender_input']
            
            # Ensure shapes are known
            pooled = tf.ensure_shape(pooled, [128, 128, 1])
            heatmap = tf.ensure_shape(heatmap, [128, 128, 1])
            gender = tf.ensure_shape(gender, [1])
            age_target = tf.ensure_shape(age_target, [])
            
            # Apply augmentation directly instead of using py_function
            if tf.random.uniform([], 0, 1) > 0.5:  # 50% chance of augmentation
                # Apply the same augmentation to both images
                pooled_aug = apply_augmentation(pooled)
                heatmap_aug = apply_augmentation(heatmap)
            else:
                pooled_aug = pooled
                heatmap_aug = heatmap
            
            # Return data in the correct format for the model
            return {
                'pooled_input': tf.cast(pooled_aug, tf.float32),
                'heatmap_input': tf.cast(heatmap_aug, tf.float32),
                'gender_input': tf.cast(gender, tf.float32)
            }, tf.cast(age_target, tf.float32)
            
        train_dataset = train_dataset.map(
            augment_map_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Apply shuffling, batching and prefetching
    train_dataset = train_dataset.shuffle(
        buffer_size=min(len(train_files), 1000),  # Use a reasonable buffer size
        reshuffle_each_iteration=True
    ).batch(
        batch_size, 
        drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)
    
    # Print dataset shapes for debugging
    print("Training dataset structure:")
    for x, y in train_dataset.take(1):
        print(f"Input shapes: {', '.join([f'{k}: {v.shape}' for k, v in x.items()])}")
        print(f"Target shape: {y.shape}")
    
    # Carica i dati di validation
    val_files = [f for f in os.listdir(val_files_path) if f.endswith('.data-00000-of-00001')]
    val_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    
    # Ensure path compatibility across platforms
    val_labels_path = os.path.normpath(os.path.join(val_files_path, "..", "val_labels.csv"))
    val_heatmaps_path = os.path.normpath(os.path.join(val_files_path, "..", "heatmaps"))
    
    print(f"Loading {len(val_files)} validation files from {val_files_path}")
    for file in val_files:
        file_prefix = os.path.join(val_files_path, file.split(".data")[0])
        try:
            id_img = int(file.split("_")[1].split(".data")[0])
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = load_tensor(file_prefix.replace("heatmaps", "tensors"), "tensor/.ATTRIBUTES/VARIABLE_VALUE")
        heatmap_tensor = load_tensor(file_prefix, "heated/.ATTRIBUTES/VARIABLE_VALUE")

        try:
            gender, age = filelabels_search(val_labels_path, id_img)
            
            # Ensure tensors have correct shapes and types
            pooled = tf.transpose(pooled_tensor, perm=[1, 2, 0])  # added transpose
            heatmap = tf.transpose(heatmap_tensor, perm=[1, 2, 0])  # added transpose
            
            # Validate tensor shapes
            if pooled.shape[0] != 128 or pooled.shape[1] != 128 or pooled.shape[2] != 1:
                print(f"Warning: Pooled tensor for validation file {file} has incorrect shape {pooled.shape}, expected (128, 128, 1)")
                continue
                
            if heatmap.shape[0] != 128 or heatmap.shape[1] != 128 or heatmap.shape[2] != 1:
                print(f"Warning: Heatmap tensor for validation file {file} has incorrect shape {heatmap.shape}, expected (128, 128, 1)")
                continue
            
            # Add to validation data
            val_data["pooled"].append(pooled)
            val_data["heatmap"].append(heatmap)
            val_data["gender"].append(gender)
            val_data["age"].append(age)
            
        except Exception as e:
            print(f"Error processing validation file {file}: {str(e)}")
            continue
    
    # Check if we have any validation data
    if len(val_data["pooled"]) == 0:
        raise ValueError("No valid validation data found. Check validation files and paths.")
        
    print(f"Loaded {len(val_data['pooled'])} validation samples successfully")
    
    # Process gender values
    try:
        val_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in val_data["gender"]]
    except Exception as e:
        print(f"Error processing gender values for validation: {str(e)}")
        # If there's an error in gender mapping, attempt to fix data
        corrected_genders = []
        for g in val_data["gender"]:
            if isinstance(g, str) and g in gender_mapping:
                corrected_genders.append(gender_mapping[g])
            elif isinstance(g, (int, float)) and g in [0, 1]:
                corrected_genders.append(g)
            else:
                print(f"Invalid gender value: {g}, defaulting to 0")
                corrected_genders.append(0)
        val_data["gender"] = corrected_genders
    
    # Stack tensors and convert to proper types with error handling
    try:
        val_pooled_tensors = tf.stack(val_data["pooled"])
        val_heatmap_tensors = tf.stack(val_data["heatmap"])
        val_gender_tensors = tf.convert_to_tensor(val_data["gender"], dtype=tf.float32)
        val_age_tensors = tf.convert_to_tensor(val_data["age"], dtype=tf.float32)
        
        # Print shape information for debugging
        print(f"Validation tensors shapes - Pooled: {val_pooled_tensors.shape}, Heatmap: {val_heatmap_tensors.shape}, Gender: {val_gender_tensors.shape}, Age: {val_age_tensors.shape}")
    except Exception as e:
        print(f"Error stacking validation tensors: {str(e)}")
        raise ValueError("Failed to create validation tensors. Check validation data format.")
    
    # Reshape gender tensor if needed
    val_gender_tensors = tf.reshape(val_gender_tensors, [-1, 1])
    # Create validation dataset with the same structure as training
    val_inputs = {
        'pooled_input': val_pooled_tensors,
        'heatmap_input': val_heatmap_tensors,
        'gender_input': val_gender_tensors
    }
    val_outputs = val_age_tensors
    
    # Create the validation dataset with (inputs, targets) structure
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_outputs))
    
    # For validation, we don't need augmentation but we should ensure the format is consistent
    def format_val_fn(inputs_dict, age_target):
        try:
            # Ensure consistent formatting with training data and set explicit shapes
            pooled = tf.ensure_shape(inputs_dict['pooled_input'], [128, 128, 1])
            heatmap = tf.ensure_shape(inputs_dict['heatmap_input'], [128, 128, 1])
            gender = tf.ensure_shape(inputs_dict['gender_input'], [1])
            age_target = tf.ensure_shape(age_target, [])
            
            # Cast to proper types
            pooled = tf.cast(pooled, tf.float32)
            heatmap = tf.cast(heatmap, tf.float32)
            gender = tf.cast(gender, tf.float32)
            age_target = tf.cast(age_target, tf.float32)
            
            return {
                'pooled_input': pooled,
                'heatmap_input': heatmap,
                'gender_input': gender
            }, age_target
        except Exception as e:
            # In case of errors, print a warning and provide valid tensors with the right shapes
            print(f"Error in validation format function: {str(e)}")
            # Return zero-filled tensors with correct shapes and types as fallback
            return {
                'pooled_input': tf.zeros([128, 128, 1], dtype=tf.float32),
                'heatmap_input': tf.zeros([128, 128, 1], dtype=tf.float32),
                'gender_input': tf.zeros([1], dtype=tf.float32)
            }, tf.constant(0.0, dtype=tf.float32)
        
    val_dataset = val_dataset.map(
        format_val_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    # Print validation dataset shapes for debugging
    print("Validation dataset structure:")
    try:
        for x, y in val_dataset.take(1):
            print(f"Input shapes: {', '.join([f'{k}: {v.shape}' for k, v in x.items()])}")
            print(f"Target shape: {y.shape}")
            # Verify data types
            print(f"Input types: {', '.join([f'{k}: {v.dtype}' for k, v in x.items()])}")
            print(f"Target type: {y.dtype}")
    except Exception as e:
        print(f"Error inspecting validation dataset: {str(e)}")
    # Creazione directory per i log e i checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creazione directory con path compatibili cross-platform
    log_dir = os.path.normpath(os.path.join("logs", timestamp))
    checkpoint_dir = os.path.normpath(os.path.join("checkpoints", timestamp))
    
    # Crea le directory se non esistono
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Filename per i checkpoint e i log (usando .keras invece di .h5)
    checkpoint_filepath = os.path.normpath(os.path.join(checkpoint_dir, "model_best.keras"))
    csv_log_filepath = os.path.normpath(os.path.join(log_dir, "training_log.csv"))
    
    # Definizione delle metriche
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_mae = tf.keras.metrics.MeanAbsoluteError(name='val_mae')
    
    # Definizione dell'ottimizzatore e della funzione di loss
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_fn = losses.MeanSquaredError()
    
    print(f"Inizializzato ottimizzatore Adam con learning rate: {learning_rate}")
    
    # Calcolo dei passi totali per il learning rate scheduler
    steps_per_epoch = len(train_files) // batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = min(1000, total_steps // 5)  # 20% degli step totali o 1000, il minore dei due
    
    # Callbacks
    print(f"Configurazione scheduler - total_steps: {total_steps}, warmup_steps: {warmup_steps}")
    lr_scheduler = WarmUpCosineDecayScheduler(
        learning_rate_base=learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )
    
    csv_logger = CSVLogger(
        csv_log_filepath,
        append=True,
        separator=','
    )
    
    callbacks_list = [
        lr_scheduler,
        model_checkpoint,
        early_stopping,
        tensorboard_callback,
        csv_logger
    ]
    
    # Tracciamento delle metriche
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_maes = []
    epoch_val_maes = []
    learning_rates = []
    
    # Preparazione plot interattivo
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MAE")
    axs[1].set_title("Training and Validation MAE")
    
    # Compilazione del modello con loss e metriche standard
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[CustomMAE()]
    )
    
    print("Inizio training...")
    
    # Utilizzo del metodo fit con callbacks e gestione errori
    try:
        # Print model summary before training
        print("\nModel Summary:")
        model.summary()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=callbacks_list,
            verbose=1
        )
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create a minimal history object to avoid errors in the plotting code
        class DummyHistory:
            def __init__(self):
                self.history = {'loss': [999.0], 'val_loss': [999.0]}
                
        history = DummyHistory()
        print("\nTraining failed. Created dummy history for plotting.")
    
    # Salvataggio del modello finale (se non è stato già salvato dal checkpoint)
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Salva il modello nel formato compatibile con TensorFlow/Keras recente
        model.save(save_path)
        print(f"Modello salvato con successo in {save_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {str(e)}")
        
        # Tenta un salvataggio alternativo
        try:
            alternative_path = os.path.join(os.path.dirname(save_path), 
                                            f"backup_model_{timestamp}.keras")
            model.save_weights(alternative_path)
            print(f"Salvati solo i pesi del modello in {alternative_path}")
        except Exception as e2:
            print(f"Impossibile salvare anche i pesi: {str(e2)}")
    
    # Aggiornamento dei dati per i plot finali
    # Aggiornamento dei dati per i plot finali con gestione degli errori
    try:
        epoch_train_losses = history.history['loss']
        epoch_val_losses = history.history['val_loss']
        epoch_train_losses = history.history['loss']
        epoch_val_losses = history.history['val_loss']
        
        # Since we're now using a more standard approach, the keys should be more predictable
        mae_keys = ['mae']
        val_mae_keys = ['val_mae']
        
        # Try to find MAE in the history using multiple possible keys
        epoch_train_maes = None
        for key in mae_keys:
            if key in history.history:
                epoch_train_maes = history.history[key]
                print(f"Found training MAE metric under key: {key}")
                break
        
        # If we still don't have MAE values, create empty ones
        if epoch_train_maes is None:
            print("Warning: No training MAE metric found in history. Using zeros.")
            epoch_train_maes = [0] * len(epoch_train_losses)
            
        # Try to find validation MAE in the history using multiple possible keys    
        epoch_val_maes = None
        for key in val_mae_keys:
            if key in history.history:
                epoch_val_maes = history.history[key]
                print(f"Found validation MAE metric under key: {key}")
                break
        
        # If we still don't have validation MAE values, create empty ones
        if epoch_val_maes is None:
            print("Warning: No validation MAE metric found in history. Using zeros.")
            epoch_val_maes = [0] * len(epoch_val_losses)
        # Verifica che learning_rates sia della stessa lunghezza degli altri dati
        if len(learning_rates) < len(epoch_train_losses):
            # Estendi learning_rates se necessario
            learning_rates.extend([learning_rates[-1]] * (len(epoch_train_losses) - len(learning_rates)))
        elif len(learning_rates) > len(epoch_train_losses):
            # Tronca learning_rates se necessario
            learning_rates = learning_rates[:len(epoch_train_losses)]
            
        print(f"Raccolte metriche per {len(epoch_train_losses)} epoche")
        
    except Exception as e:
        print(f"Errore durante la raccolta delle metriche di training: {str(e)}")
        # Crea dati vuoti per evitare errori nei grafici
        num_epochs_completed = len(getattr(history.history, 'loss', []))
        if num_epochs_completed == 0:
            num_epochs_completed = 1  # Fallback
            
        epoch_train_losses = [0] * num_epochs_completed
        epoch_val_losses = [0] * num_epochs_completed
        epoch_train_maes = [0] * num_epochs_completed
        epoch_val_maes = [0] * num_epochs_completed
        learning_rates = [learning_rate] * num_epochs_completed
    
    # Plot finali
    plt.figure(figsize=(15, 15))
    
    # Plot delle loss
    plt.subplot(3, 1, 1)
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot del MAE
    plt.subplot(3, 1, 2)
    plt.plot(epoch_train_maes, label='Training MAE')
    plt.plot(epoch_val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (months)')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    
    # Plot del learning rate
    plt.subplot(3, 1, 3)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    # Usa path normalizzato per compatibilità cross-platform
    plot_path = os.path.normpath(os.path.join(log_dir, "training_plots.png"))
    plt.savefig(plot_path)
    plt.show()
    
    # Preparazione dell'oggetto risultato da ritornare
    training_history = {
        'train_loss': epoch_train_losses,
        'val_loss': epoch_val_losses,
        'train_mae': epoch_train_maes,
        'val_mae': epoch_val_maes,
        'learning_rates': learning_rates,
        'best_val_loss': min(epoch_val_losses),
        'best_val_mae': min(epoch_val_maes),
        'best_epoch': np.argmin(epoch_val_losses) + 1,
        'total_epochs': len(epoch_val_losses)
    }
    
    # Stampa delle statistiche finali
    print("\nStatistiche di training:")
    print(f"- Miglior validation loss: {training_history['best_val_loss']:.4f} (epoch {training_history['best_epoch']})")
    print(f"- Miglior validation MAE: {training_history['best_val_mae']:.4f} mesi")
    print(f"- Numero di epoche completate: {training_history['total_epochs']}")
    
    # Se early stopping è stato attivato
    if training_history['total_epochs'] < num_epochs:
        print(f"- Early stopping attivato dopo {training_history['total_epochs']} epoche")
    
    return training_history
