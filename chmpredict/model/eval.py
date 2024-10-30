import os
import torch


def eval_fn(loader, model, criterion, output_dir, device):
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    test_loss = eval_loop(loader, model, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")


def eval_loop(loader, model, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(loader)
    return avg_val_loss
