import os
import torch

from tqdm import tqdm

from chmpredict.model.eval import eval_loop


def train_fn(train_loader, val_loader, model, criterion, optimizer, num_epochs, patience, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        
        val_loss = eval_loop(val_loader, model, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
            early_stopping_counter = 0
            print(f"[INFO] Validation loss improved. Model saved at epoch {epoch + 1}.")
        else:
            early_stopping_counter += 1
            print(f"[INFO] Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break


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