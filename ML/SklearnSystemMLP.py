import json
import os
import copy
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor


class SklearnSystemMLP:
    """
    Model MLP oparty o scikit-learn do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))
    """

    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(self.CURRENT_DIR)

        self.training_config = {}
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "total_epochs": 0
        }

        self.best_model_state = None

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1,          # jedna epoka na jedno fit()
            warm_start=True,     # kontynuacja między epokami
            shuffle=True,
            random_state=seed,
            early_stopping=False,
            tol=0.0
        )

    @staticmethod
    def mse_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def predict(self, X):
        return self.model.predict(X)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        lr=0.001,
        epochs=100,
        patience=10,
        verbose=True,
    ):
        """
        Trening podobny logiką do Twojego SystemMLP / OwnSystemMLP.
        Uwaga: sklearn nie daje tak niskopoziomowej kontroli jak torch,
        więc trenujemy epoka po epoce przez warm_start=True i max_iter=1.
        """
        self.training_config = {
            "lr": lr,
            "max_epochs": epochs,
            "patience": patience,
            "samples_train": X_train.shape[0],
            "optimizer": "Adam",
            "library": "scikit-learn"
        }

        self.training_history["train_loss"] = []
        self.training_history["val_loss"] = []
        self.training_history["lr"] = []
        self.training_history["best_epoch"] = 0
        self.training_history["total_epochs"] = 0

        # ustaw LR przy starcie treningu
        self.model.learning_rate_init = lr

        if verbose:
            print(f"\nSKLEARN MLP - Start treningu: {epochs} epok, {X_train.shape[0]} trajektorii na epokę.")

        # spłaszczamy dane treningowe i walidacyjne do [N x cechy]
        if X_train.ndim == 3:
            X_train_fit = X_train.reshape(-1, X_train.shape[-1])
            y_train_fit = y_train.reshape(-1, y_train.shape[-1])
        else:
            X_train_fit = X_train
            y_train_fit = y_train

        if X_val.ndim == 3:
            X_val_eval = X_val.reshape(-1, X_val.shape[-1])
            y_val_eval = y_val.reshape(-1, y_val.shape[-1])
        else:
            X_val_eval = X_val
            y_val_eval = y_val

        best_val_loss = float("inf")
        epochs_no_improve = 0

        epoch_bar = tqdm(range(epochs), desc="Trening Sklearn MLP", unit="epoka")

        for epoch in epoch_bar:
            self.model.fit(X_train_fit, y_train_fit)

            y_train_pred = self.model.predict(X_train_fit)
            avg_train_loss = self.mse_loss(y_train_pred, y_train_fit)

            v_pred = self.model.predict(X_val_eval)
            v_loss = self.mse_loss(v_pred, y_val_eval)

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(v_loss)
            self.training_history["lr"].append(lr)
            self.training_history["total_epochs"] = epoch + 1

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(self.model)
                self.training_history["best_epoch"] = epoch + 1
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix({
                "T_Loss": f"{avg_train_loss:.8f}",
                "V_Loss": f"{v_loss:.8f}",
                "Patience": f"{epochs_no_improve}/{patience}"
            })

            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"🛑 Early Stopping! Brak poprawy przez {patience} epok. "
                        f"Aktualna epoka: {epoch + 1}"
                    )
                break

        if self.best_model_state is not None:
            self.model = copy.deepcopy(self.best_model_state)
            if verbose:
                print(
                    f"✅ Przywrócono najlepszy model "
                    f"(Val MSE: {best_val_loss:.8f}) z epoki {self.training_history['best_epoch']}"
                )

    def simulate(self, t, u_new, h0, dh_dt0=None):
        n_points = len(t)
        dt = t[1] - t[0]

        h_sim = np.zeros((n_points, self.output_dim))
        dh_dt_sim = np.zeros((n_points, self.output_dim))

        h_sim[0] = h0

        if dh_dt0 is None:
            dh_dt_prev = np.zeros(self.output_dim)
        else:
            dh_dt_prev = np.array(dh_dt0, dtype=float)

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
            ).astype(float).reshape(1, -1)

            dh_dt_curr = self.predict(x_input).flatten()
            dh_dt_sim[i] = dh_dt_curr

            h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt
            h_sim[i] = np.maximum(h_sim[i], 0.0)

            dh_dt_prev = dh_dt_curr

        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)

    def save_metadata(self, filepath):
        metadata = {
            "model_arch": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seed": self.seed
            },
            "training_config": self.training_config,
            "training_history": self.training_history
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def save_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        full_dir = os.path.join(self.BASE_DIR, folder, dataset)
        os.makedirs(full_dir, exist_ok=True)

        base_path = os.path.join(full_dir, base_name)

        joblib.dump(self.model, f"{base_path}.joblib")
        self.save_metadata(f"{base_path}_info.json")

        print(f"✅ Model i metadane zapisane w: {full_dir} jako {base_name}")

    def load_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        base_path = os.path.join(self.BASE_DIR, folder, dataset, base_name)
        metadata_path = f"{base_path}_info.json"
        model_path = f"{base_path}.joblib"

        try:
            self.model = joblib.load(model_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.hidden_dim = metadata["model_arch"]["hidden_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.seed = metadata["model_arch"]["seed"]
            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            self.best_model_state = copy.deepcopy(self.model)

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
            print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")