import torch
import numpy as np

from tqdm import tqdm

from chmpredict.model.build import load_best_model


def eval_fn(loader, model, criterion, output_dir, device):
    load_best_model(model, output_dir, device)
    
    metrics = eval_loop(loader, model, criterion, device)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


def eval_loop(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    metric_sums = {name: 0 for name in METRICS.keys()}  # Initialize sums for each metric

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating", unit="batch"):
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            batch_size = targets.size(0)

            total_loss += criterion(predictions, targets).item() * batch_size
            n_samples += batch_size

            for name, func in METRICS.items():
                metric_sums[name] += func(targets.cpu().numpy(), predictions.cpu().numpy())

    avg_loss = total_loss / n_samples
    avg_metrics = {name: value / n_samples for name, value in metric_sums.items()}
    avg_metrics["mse"] = avg_loss
    
    return avg_metrics


def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true))


def root_mean_squared_error(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) ** 0.5


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    non_zero = y_true != 0
    return np.sum(np.abs((y_true[non_zero] - y_pred[non_zero]) / (y_true[non_zero] + epsilon))) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    return np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-6))

def correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    cov = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    return cov / (std_true * std_pred + 1e-6)

METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "mape": mean_absolute_percentage_error,
    "smape": symmetric_mean_absolute_percentage_error,
    "r2": r2_score,
    "corr_coeff": correlation_coefficient,
}