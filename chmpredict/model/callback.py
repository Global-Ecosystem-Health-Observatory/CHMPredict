import os
import torch


class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.stop_training = True


class ModelCheckpoint(Callback):
    def __init__(self, output_dir, monitor="val_loss"):
        self.output_dir = output_dir
        self.monitor = monitor
        self.best_score = float("inf")
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        
        if score < self.best_score:
            self.best_score = score
            checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(logs["model"].state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1} with {self.monitor} = {score:.4f}")
