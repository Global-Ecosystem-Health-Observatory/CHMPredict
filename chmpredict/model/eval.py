import torch
import numpy as np

from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from chmpredict.model.build import load_best_model


def eval_fn(loader, model, criterion, output_dir, device):
    load_best_model(model, output_dir, device)
    
    test_loss, mae, rmse, mape, smape, r2, corr_coeff = eval_loop(loader, model, criterion, device)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.4f}%")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Correlation Coefficient (r): {corr_coeff:.4f}")


def eval_loop(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(loader, unit="batch", desc="Evaluating") as tepoch:
            for data, targets in tepoch:
                data, targets = data.to(device), targets.to(device)

                predictions = model(data)
                loss = criterion(predictions, targets)
                total_loss += loss.item()

                all_targets.extend(targets.cpu().numpy().flatten())
                all_predictions.extend(predictions.cpu().numpy().flatten())

                tepoch.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(np.mean((np.array(all_targets) - np.array(all_predictions)) ** 2))

    epsilon = 1e-6
    targets_array = np.array(all_targets)
    predictions_array = np.array(all_predictions)
    
    non_zero_targets = targets_array != 0
    mape = np.mean(np.abs((targets_array[non_zero_targets] - predictions_array[non_zero_targets]) 
                          / (targets_array[non_zero_targets] + epsilon))) * 100

    smape = np.mean(2 * np.abs(predictions_array - targets_array) 
                    / (np.abs(targets_array) + np.abs(predictions_array) + epsilon)) * 100
    r2 = r2_score(all_targets, all_predictions)
    corr_coeff, _ = pearsonr(all_targets, all_predictions)
    
    return avg_loss, mae, rmse, mape, smape, r2, corr_coeff