# Custom callback per il learning rate warmup seguito da cosine decay
import math
import tensorflow as tf
from keras import callbacks


class WarmUpCosineDecayScheduler(callbacks.Callback):
    def __init__(
            self,
            learning_rate_base,
            total_steps,
            warmup_steps,
            hold_base_rate_steps=0,
            verbose=0
    ):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
        self.current_lr = learning_rate_base  # Tracking current LR for logging

    def compute_lr(self, step):
        if step < self.warmup_steps:
            # Warmup phase
            lr = self.learning_rate_base * (step / self.warmup_steps)
        else:
            # Cosine decay phase
            step = min(step, self.total_steps)
            cosine_decay_steps = self.total_steps - self.warmup_steps - self.hold_base_rate_steps
            step_adjusted = step - self.warmup_steps - self.hold_base_rate_steps

            if step_adjusted < 0:
                # Still in hold phase after warmup
                return self.learning_rate_base

            # Cosine decay formula
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step_adjusted / cosine_decay_steps))
            lr = self.learning_rate_base * cosine_decay

        return lr

    def on_batch_begin(self, batch, logs=None):
        try:
            # Get optimizer's iterations as an integer
            if hasattr(self.model.optimizer, 'iterations'):
                # Try to get the value directly if possible
                if hasattr(self.model.optimizer.iterations, 'numpy'):
                    global_step = self.model.optimizer.iterations.numpy()
                else:
                    # Fall back to eval/getting python value
                    global_step = int(self.model.optimizer.iterations)
            else:
                # If iterations not available, estimate from batch
                global_step = self.model._train_counter.numpy()

            new_lr = self.compute_lr(global_step)

            # Update optimizer learning rate using the appropriate method for newer TensorFlow versions
            if hasattr(self.model.optimizer, 'learning_rate'):
                # Get the learning rate variable
                lr_var = self.model.optimizer.learning_rate
                if hasattr(lr_var, '_variable'):
                    # Access the underlying variable if it's a LearningRateSchedule
                    lr_var = lr_var._variable
                # Set the learning rate value
                if hasattr(lr_var, 'assign'):
                    lr_var.assign(new_lr)
                else:
                    # Fall back to the backend method
                    tf.keras.backend.set_value(lr_var, new_lr)
            else:
                # If learning_rate attribute doesn't exist, log a warning
                print(
                    f"\nWarning: Unable to set learning rate - optimizer doesn't have accessible 'learning_rate' attribute")
                return

            self.current_lr = new_lr  # Store for epoch_end

            if self.verbose > 0 and batch % 100 == 0:  # Reduced verbosity
                print(f"\nBatch {batch}: Learning rate set to {new_lr:.6f}")

        except Exception as e:
            print(f"\nError in learning rate scheduler: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_epoch_end(self, epoch, logs=None):
        # Store current learning rate for plotting
        try:
            # Try to get learning rate from the optimizer
            if hasattr(self.model.optimizer, 'learning_rate'):
                lr_var = self.model.optimizer.learning_rate
                # Handle different types of learning rate objects
                if hasattr(lr_var, 'numpy'):
                    lr_value = lr_var.numpy()
                elif hasattr(lr_var, '_variable') and hasattr(lr_var._variable, 'numpy'):
                    lr_value = lr_var._variable.numpy()
                else:
                    # Last resort: use the stored value
                    lr_value = self.current_lr
            else:
                # If attributes aren't available, use stored value
                lr_value = self.current_lr

            # Append the learning rate to our list for plotting
            self.learning_rates.append(float(lr_value))

            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Final learning rate was {float(lr_value):.6f}")

        except Exception as e:
            print(f"\nError logging learning rate at epoch end: {str(e)}")
            import traceback
            traceback.print_exc()
            # Still append something to maintain array length consistency
            self.learning_rates.append(float(self.current_lr))