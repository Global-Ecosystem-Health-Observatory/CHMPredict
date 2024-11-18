from tqdm import tqdm

from chmpredict.model.eval import eval_loop
from chmpredict.model.callback import EarlyStopping


def train_fn(train_loader, val_loader, model, criterion, optimizer, num_epochs, patience, output_dir, mean_chm, std_chm, device, callbacks=None):
    logs = {"model": model}

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        model.train()
        train_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        val_metrics = eval_loop(val_loader, model, criterion, mean_chm, std_chm, device)
        val_loss = val_metrics["mse"]  # Set main validation metric as `mse` for callbacks
        
        logs["val_loss"] = val_loss  # Primary metric
        logs.update(val_metrics)
        logs["epoch"] = epoch

        if callbacks:
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)
                if isinstance(callback, EarlyStopping) and callback.stop_training:
                    print("Training stopped early due to EarlyStopping.")
                    return

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss (MSE): {val_loss:.4f}")


def train_loop(loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0

    with tqdm(loader, unit="batch") as tepoch:
        for data, targets in tepoch:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, targets)
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(loader)
    return avg_train_loss