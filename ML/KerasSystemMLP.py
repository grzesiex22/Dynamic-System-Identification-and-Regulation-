import json
import os
import numpy as np
from tqdm import tqdm
import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class KerasSystemMLP:
    """
    Model MLP do identyfikacji dynamiki systemów nieliniowych.
    Przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))

    Wersja zoptymalizowana pod testowanie:
    - brak model.predict() w pętli
    - bezpośrednie wywołanie modelu: model(x, training=False)
    - dodatkowo simulate_batch() dla wielu trajektorii naraz
    """

    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        np.random.seed(seed)
        tf.random.set_seed(seed)

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

        self.model = self._build_model()
        self._compile_model(lr=0.001)

        # Szybka ścieżka inferencji
        self._fast_infer = tf.function(
            lambda x: self.model(x, training=False),
            reduce_retracing=True
        )

    def _build_model(self):
        return keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_dim, activation="tanh"),
            layers.Dense(self.hidden_dim, activation="tanh"),
            layers.Dense(self.output_dim, activation="linear"),
        ])

    def _compile_model(self, lr=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse"
        )

    def predict(self, X):
        """
        Szybsza inferencja niż model.predict(...) przy małych batchach.
        """
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        y = self._fast_infer(X)
        return y.numpy()

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
        Trening po trajektoriach, podobnie jak w Twoim Torch MLP.
        """
        self.training_config = {
            "lr": lr,
            "max_epochs": epochs,
            "patience": patience,
            "samples_train": X_train.shape[0],
            "optimizer": "Adam"
        }

        self.training_history["train_loss"] = []
        self.training_history["val_loss"] = []
        self.training_history["lr"] = []
        self.training_history["best_epoch"] = 0
        self.training_history["total_epochs"] = 0

        self.best_model_weights = None
        self._compile_model(lr=lr)

        if verbose:
            print(f"\nKERAS MLP - Start treningu: {epochs} epok, {X_train.shape[0]} trajektorii na epokę.")

        if X_val.ndim == 3:
            X_val_eval = X_val.reshape(-1, X_val.shape[-1]).astype(np.float32)
            y_val_eval = y_val.reshape(-1, y_val.shape[-1]).astype(np.float32)
        else:
            X_val_eval = X_val.astype(np.float32)
            y_val_eval = y_val.astype(np.float32)

        best_val_loss = float("inf")
        epochs_no_improve = 0

        epoch_bar = tqdm(range(epochs), desc="Trening Keras MLP", unit="epoka")

        for epoch in epoch_bar:
            train_losses = []

            for i in range(X_train.shape[0]):
                x_traj = X_train[i].astype(np.float32)
                y_traj = y_train[i].astype(np.float32)

                loss = self.model.train_on_batch(x_traj, y_traj)
                train_losses.append(float(loss))

            avg_train_loss = float(np.mean(train_losses))

            # walidacja
            v_pred = self.predict(X_val_eval)
            v_loss = float(np.mean((v_pred - y_val_eval) ** 2))

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(v_loss)
            self.training_history["lr"].append(lr)
            self.training_history["total_epochs"] = epoch + 1

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                self.best_model_weights = copy.deepcopy(self.model.get_weights())
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
                    print(f"🛑 Early Stopping! Brak poprawy przez {patience} epok. Aktualna epoka: {epoch}")
                break

        if self.best_model_weights is not None:
            self.model.set_weights(self.best_model_weights)
            if verbose:
                print(
                    f"✅ Przywrócono najlepszy model "
                    f"(Val MSE: {best_val_loss:.8f}) z epoki {self.training_history['best_epoch']}"
                )

    def simulate(self, t, u_new, h0, dh_dt0=None):
        """
        Symulacja pojedynczej trajektorii.
        Nadal rekurencyjna po czasie, ale bez model.predict().
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

        u_new = np.asarray(u_new, dtype=np.float32)

        for i in range(1, n_points):
            x_input = np.concatenate([
                u_new[i],
                h_sim[i - 1],
                dh_dt_prev,
            ], axis=0).reshape(1, -1)

            dh_dt_curr = self.predict(x_input).reshape(-1)
            dh_dt_sim[i] = dh_dt_curr

            h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt
            h_sim[i] = np.maximum(h_sim[i], 0.0)

            dh_dt_prev = dh_dt_curr

        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)

    def simulate_batch(self, t, u_batch, h0_batch, dh_dt0_batch=None):
        """
        Symulacja wielu trajektorii naraz.

        Args:
            t: [T]
            u_batch: [B, T, dim_u] lub [B, T]
            h0_batch: [B, output_dim]
            dh_dt0_batch: [B, output_dim] lub None

        Returns:
            tuple: (h_sim, dh_dt_sim)
                h_sim: [B, T, output_dim]
                dh_dt_sim: [B, T, output_dim]
        """
        u_batch = np.asarray(u_batch, dtype=np.float32)
        h0_batch = np.asarray(h0_batch, dtype=np.float32)

        if u_batch.ndim == 2:
            u_batch = u_batch[..., np.newaxis]

        B, T, _ = u_batch.shape
        dt = t[1] - t[0]

        h_sim = np.zeros((B, T, self.output_dim), dtype=np.float32)
        dh_dt_sim = np.zeros((B, T, self.output_dim), dtype=np.float32)

        h_sim[:, 0, :] = h0_batch

        if dh_dt0_batch is None:
            dh_prev = np.zeros((B, self.output_dim), dtype=np.float32)
        else:
            dh_prev = np.asarray(dh_dt0_batch, dtype=np.float32)

        dh_dt_sim[:, 0, :] = dh_prev

        for i in range(1, T):
            x_input = np.concatenate([
                u_batch[:, i, :],
                h_sim[:, i - 1, :],
                dh_prev
            ], axis=1)

            dh_curr = self.predict(x_input)
            dh_dt_sim[:, i, :] = dh_curr

            h_sim[:, i, :] = h_sim[:, i - 1, :] + dh_curr * dt
            h_sim[:, i, :] = np.maximum(h_sim[:, i, :], 0.0)

            dh_prev = dh_curr

        return h_sim, dh_dt_sim

    def save_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        full_dir = os.path.join(self.BASE_DIR, folder, dataset)
        os.makedirs(full_dir, exist_ok=True)

        base_path = os.path.join(full_dir, base_name)

        self.model.save(f"{base_path}.keras")

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

        with open(f"{base_path}_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"✅ Model i metadane zapisane w: {full_dir} jako {base_name}")

    def load_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        base_path = os.path.join(self.BASE_DIR, folder, dataset, base_name)
        metadata_path = f"{base_path}_info.json"
        model_path = f"{base_path}.keras"

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.hidden_dim = metadata["model_arch"]["hidden_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.seed = metadata["model_arch"]["seed"]
            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

            self.model = keras.models.load_model(model_path)

            # po wczytaniu odtwarzamy szybką ścieżkę inferencji
            self._fast_infer = tf.function(
                lambda x: self.model(x, training=False),
                reduce_retracing=True
            )

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
            print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")