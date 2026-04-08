import numpy as np


class Metrics:
    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(Metrics.mse(y_true, y_pred)))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "mse": Metrics.mse(y_true, y_pred),
            "rmse": Metrics.rmse(y_true, y_pred),
            "mae": Metrics.mae(y_true, y_pred),
            "r2": Metrics.r2(y_true, y_pred),
        }

    def print_metrics(title, metrics):
        print(f"\n=== {title} ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.8f}")