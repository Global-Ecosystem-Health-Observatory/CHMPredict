from tqdm import tqdm

from chmpredict.model.eval import eval_loop


def train_fn(train_loader, val_loader, model, criterion, optimizer, num_epochs, patience, output_dir, device, callbacks=None):
    if callbacks is None:
        callbacks = []

    for callback in callbacks:
        callback.on_train_begin()

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        train_loss = train_loop(train_loader, model, criterion, optimizer, device)

        val_loss = eval_loop(val_loader, model, criterion, device)
        logs = {"val_loss": val_loss, "train_loss": train_loss, "model": model}

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if any(getattr(callback, "stop_training", False) for callback in callbacks):
            print("Training stopped by a callback.")
            break

    for callback in callbacks:
        callback.on_train_end()


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

            # Update progress bar with the current loss
            tepoch.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(loader)
    return avg_train_loss