import torch
import numpy as np

from tqdm import tqdm

from chmpredict.model.build import load_best_model


def eval_fn(loader, model, criterion, output_dir, mean_chm, std_chm, device):
    load_best_model(model, output_dir, device)
    
    metrics = eval_loop(loader, model, criterion, mean_chm, std_chm, device)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


def eval_loop(loader, model, criterion, mean_chm, std_chm, device, nan_value=-9999):
    model.eval()
    total_loss, total_mae, total_rmse = 0, 0, 0
    total_mape, total_smape = 0, 0
    epsilon = 1e-6  # To handle divide-by-zero
    n_samples = 0
    
    y_true_sum, y_pred_sum = 0, 0
    y_true_sq_sum, y_pred_sq_sum = 0, 0
    y_pred_y_true_sum = 0
    
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating", unit="batch"):
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            
            batch_size = targets.size(0)
            total_loss += criterion(predictions, targets).item() * batch_size
            
            n_samples += batch_size

            predictions = predictions * std_chm + mean_chm
            targets = targets * std_chm + mean_chm
            
            mask = targets != nan_value
            targets = targets[mask]
            predictions = predictions[mask]

            targets_np = targets.cpu().numpy().flatten()
            predictions_np = predictions.cpu().numpy().flatten()

            total_mae += np.sum(np.abs(predictions_np - targets_np))
            total_rmse += np.sum((predictions_np - targets_np) ** 2)

            min_height_threshold = 1.0  # Threshold for filtering near-zero target values
            valid_mape_smape = targets_np > min_height_threshold
            total_mape += np.sum(np.abs((predictions_np[valid_mape_smape] - targets_np[valid_mape_smape]) 
                                        / targets_np[valid_mape_smape])) * 100
            total_smape += np.sum(2 * np.abs(predictions_np[valid_mape_smape] - targets_np[valid_mape_smape]) 
                                  / (np.abs(targets_np[valid_mape_smape]) + np.abs(predictions_np[valid_mape_smape]) + epsilon)) * 100

            y_true_sum += np.sum(targets_np)
            y_pred_sum += np.sum(predictions_np)
            y_true_sq_sum += np.sum(targets_np ** 2)
            y_pred_sq_sum += np.sum(predictions_np ** 2)
            y_pred_y_true_sum += np.sum(predictions_np * targets_np)
    
    avg_loss = total_loss / n_samples
    mae = total_mae / n_samples

    rmse = np.sqrt(total_rmse / n_samples)

    mape = total_mape / n_samples
    smape = total_smape / n_samples

    ss_res = np.sum((predictions_np - targets_np) ** 2)

    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + epsilon))

    numerator = n_samples * y_pred_y_true_sum - y_pred_sum * y_true_sum
    denominator = np.sqrt((n_samples * y_pred_sq_sum - y_pred_sum ** 2) * 
                        (n_samples * y_true_sq_sum - y_true_sum ** 2))
    corr_coeff = numerator / (denominator + epsilon)

    return {"mse": avg_loss, "mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2, "corr_coeff": corr_coeff}