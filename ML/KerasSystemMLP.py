import json
import os
import copy
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class KerasSystemMLP:
    """
    Model MLP oparty o TensorFlow / Keras do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))
    """

    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        tf.keras.utils.set_random_seed(seed)

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

        self.best_model_weights = None

        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_dim, activation="tanh"),
            layers.Dense(hidden_dim, activation="tanh"),
            layers.Dense(output_dim, activation="linear")
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError()
        )

    @staticmethod
    def mse_loss(y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def predict(self, X):
        return self.model.predict(X, verbose=0)

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
        self.training_config = {
            "lr": lr,
            "max_epochs": epochs,
            "patience": patience,
            "samples_train": X_train.shape[0],
            "optimizer": "Adam",
            "library": "TensorFlow/Keras"
        }

        self.training_history["train_loss"] = []
        self.training_history["val_loss"] = []
        self.training_history["lr"] = []
        self.training_history["best_epoch"] = 0
        self.training_history["total_epochs"] = 0

        if verbose:
            print(f"\nKERAS MLP - Start treningu: {epochs} epok, {X_train.shape[0]} trajektorii na epokę.")

        # Spłaszczamy trajektorie do klasycznego supervised learning [N x cechy]
        if X_train.ndim == 3:
            X_train_fit = X_train.reshape(-1, X_train.shape[-1]).astype(np.float32)
            y_train_fit = y_train.reshape(-1, y_train.shape[-1]).astype(np.float32)
        else:
            X_train_fit = X_train.astype(np.float32)
            y_train_fit = y_train.astype(np.float32)

        if X_val.ndim == 3:
            X_val_fit = X_val.reshape(-1, X_val.shape[-1]).astype(np.float32)
            y_val_fit = y_val.reshape(-1, y_val.shape[-1]).astype(np.float32)
        else:
            X_val_fit = X_val.astype(np.float32)
            y_val_fit = y_val.astype(np.float32)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0

        epoch_bar = tqdm(range(epochs), desc="Trening Keras MLP", unit="epoka")

        for epoch in epoch_bar:
            hist = self.model.fit(
                X_train_fit,
                y_train_fit,
                validation_data=(X_val_fit, y_val_fit),
                epochs=1,
                batch_size=256,
                verbose=0,
                shuffle=True
            )

            train_loss = float(hist.history["loss"][0])
            val_loss = float(hist.history["val_loss"][0])

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["lr"].append(lr)
            self.training_history["total_epochs"] = epoch + 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_model_weights = copy.deepcopy(self.model.get_weights())
                self.training_history["best_epoch"] = epoch + 1
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix({
                "T_Loss": f"{train_loss:.8f}",
                "V_Loss": f"{val_loss:.8f}",
                "Patience": f"{epochs_no_improve}/{patience}"
            })

            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"🛑 Early Stopping! Brak poprawy przez {patience} epok. "
                        f"Aktualna epoka: {epoch + 1}"
                    )
                break

        if self.best_model_weights is not None:
            self.model.set_weights(self.best_model_weights)
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
            ).astype(np.float32).reshape(1, -1)

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

        self.model.save(f"{base_path}.keras")
        self.save_metadata(f"{base_path}_info.json")

        print(f"✅ Model i metadane zapisane w: {full_dir} jako {base_name}")

    def load_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        base_path = os.path.join(self.BASE_DIR, folder, dataset, base_name)
        metadata_path = f"{base_path}_info.json"
        model_path = f"{base_path}.keras"

        try:
            self.model = keras.models.load_model(model_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.hidden_dim = metadata["model_arch"]["hidden_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.seed = metadata["model_arch"]["seed"]
            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            self.best_model_weights = copy.deepcopy(self.model.get_weights())

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
            print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")