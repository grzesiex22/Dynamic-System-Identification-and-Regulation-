import json
import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge


class RegressionSystemModel:
    """
    Model regresyjny typu Ridge do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))
    """

    def __init__(self, input_dim=5, output_dim=2, alpha=1.0, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.seed = seed

        self.model = Ridge(alpha=self.alpha)

        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(self.CURRENT_DIR)

        self.training_config = {}
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 1,
            "total_epochs": 1
        }

    @staticmethod
    def mse_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        lr=0.001,
        epochs=1,
        patience=1,
        verbose=True,
    ):
        """
        Trening baseline'u regresyjnego.
        Parametry lr / epochs / patience zostawione tylko dla zgodności interfejsu.
        """
        if X_train.ndim == 3:
            X_train_fit = X_train.reshape(-1, X_train.shape[-1]).astype(np.float32)
            y_train_fit = y_train.reshape(-1, y_train.shape[-1]).astype(np.float32)
        else:
            X_train_fit = np.asarray(X_train, dtype=np.float32)
            y_train_fit = np.asarray(y_train, dtype=np.float32)

        if X_val.ndim == 3:
            X_val_eval = X_val.reshape(-1, X_val.shape[-1]).astype(np.float32)
            y_val_eval = y_val.reshape(-1, y_val.shape[-1]).astype(np.float32)
        else:
            X_val_eval = np.asarray(X_val, dtype=np.float32)
            y_val_eval = np.asarray(y_val, dtype=np.float32)

        self.training_config = {
            "alpha": self.alpha,
            "samples_train": int(X_train_fit.shape[0]),
            "optimizer": "closed_form/solver_internal",
            "model_type": "RidgeRegression"
        }

        if verbose:
            print(f"\nRIDGE REGRESSOR - Start treningu na {X_train_fit.shape[0]} próbkach.")

        self.model.fit(X_train_fit, y_train_fit)

        train_pred = self.model.predict(X_train_fit)
        val_pred = self.model.predict(X_val_eval)

        train_loss = self.mse_loss(train_pred, y_train_fit)
        val_loss = self.mse_loss(val_pred, y_val_eval)

        self.training_history["train_loss"] = [train_loss]
        self.training_history["val_loss"] = [val_loss]
        self.training_history["lr"] = [0.0]
        self.training_history["best_epoch"] = 1
        self.training_history["total_epochs"] = 1

        if verbose:
            print(f"✅ Trening zakończony | Train MSE: {train_loss:.8f} | Val MSE: {val_loss:.8f}")

    def simulate(self, t, u_new, h0, dh_dt0=None):
        """
        Symulacja rekurencyjna (Open-Loop) systemu przy użyciu modelu regresyjnego.
        """
        n_points = len(t)
        dt = t[1] - t[0]

        h_sim = np.zeros((n_points, self.output_dim), dtype=np.float32)
        dh_dt_sim = np.zeros((n_points, self.output_dim), dtype=np.float32)

        h_sim[0] = np.asarray(h0, dtype=np.float32)

        if dh_dt0 is None:
            dh_dt_prev = np.zeros(self.output_dim, dtype=np.float32)
        else:
            dh_dt_prev = np.asarray(dh_dt0, dtype=np.float32)

        dh_dt_sim[0] = dh_dt_prev

        if u_new.ndim == 1:
            u_new = u_new.reshape(-1, 1)

        for i in range(1, n_points):
            x_input = np.concatenate(
                [
                    u_new[i],
                    h_sim[i - 1],
                    dh_dt_prev,
                ]
            ).astype(np.float32).reshape(1, -1)

            dh_dt_curr = self.predict(x_input).flatten().astype(np.float32)
            dh_dt_sim[i] = dh_dt_curr

            h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt
            h_sim[i] = np.maximum(h_sim[i], 0.0)

            dh_dt_prev = dh_dt_curr

        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)

    def save_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        full_dir = os.path.join(self.BASE_DIR, folder, dataset)
        os.makedirs(full_dir, exist_ok=True)

        base_path = os.path.join(full_dir, base_name)

        with open(f"{base_path}.pkl", "wb") as f:
            pickle.dump(self.model, f)

        metadata = {
            "model_arch": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "alpha": self.alpha,
                "seed": self.seed
            },
            "training_config": self.training_config,
            "training_history": self.training_history
        }

        with open(f"{base_path}_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"✅ Model i metadane zapisane w: {full_dir} jako {base_name}")

    def load_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        base_path = os.path.join(self.BASE_DIR, folder, dataset, base_name)
        metadata_path = f"{base_path}_info.json"
        model_path = f"{base_path}.pkl"

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.alpha = metadata["model_arch"]["alpha"]
            self.seed = metadata["model_arch"]["seed"]
            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
            print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")