import numpy as np
import copy
import os
from tqdm import tqdm


class OwnSystemMLP:
    """
    Model sieci neuronowej MLP do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))
    """

    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, seed=42):
        """
        Inicjalizacja architektury sieci (Xavier initialization).

        Args:
            input_dim (int): Liczba cech wejściowych.
            hidden_dim (int): Liczba neuronów w warstwach ukrytych.
            output_dim (int): Liczba wyjść.
            seed (int): Ziarno losowości.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        rng = np.random.default_rng(seed)

        def xavier_init(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        # Xavier initialization
        self.W1 = xavier_init(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = xavier_init(hidden_dim, hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))

        self.W3 = xavier_init(hidden_dim, output_dim)
        self.b3 = np.zeros((1, output_dim))

        self.best_model_state = None
        self.best_epoch_nr = 0

        # Stany optymalizatora Adam
        self.m = {}
        self.v = {}
        self._init_optimizer_states()

    def _init_optimizer_states(self):
        """Inicjalizacja stanów optymalizatora Adam."""
        self.m = {}
        self.v = {}
        for name, param in self.get_parameters().items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def get_parameters(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def set_parameters(self, params):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        self.W3 = params["W3"].copy()
        self.b3 = params["b3"].copy()

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_derivative(a):
        # a = tanh(z)
        return 1.0 - a**2

    def forward(self, X):
        """
        Forward pass.

        Args:
            X (np.ndarray): [N x input_dim]

        Returns:
            tuple: (y_pred, cache)
        """
        z1 = X @ self.W1 + self.b1
        a1 = self._tanh(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self._tanh(z2)

        z3 = a2 @ self.W3 + self.b3
        y_pred = z3

        cache = {
            "X": X,
            "a1": a1,
            "a2": a2,
            "y_pred": y_pred,
        }
        return y_pred, cache

    @staticmethod
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true, cache):
        """
        Backpropagation dla MSE.

        Args:
            y_true (np.ndarray): [N x output_dim]
            cache (dict): dane z forward

        Returns:
            dict: gradienty wag i biasów
        """
        X = cache["X"]
        a1 = cache["a1"]
        a2 = cache["a2"]
        y_pred = cache["y_pred"]

        N = X.shape[0]

        # dL/dy_pred
        dy = (2.0 / N) * (y_pred - y_true)

        # Warstwa 3
        dW3 = a2.T @ dy
        db3 = np.sum(dy, axis=0, keepdims=True)

        da2 = dy @ self.W3.T
        dz2 = da2 * self._tanh_derivative(a2)

        # Warstwa 2
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._tanh_derivative(a1)

        # Warstwa 1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }
        return grads

    def adam_step(self, grads, lr, step_idx, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Jedna aktualizacja wag algorytmem Adam.
        """
        params = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }

        for name in params:
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grads[name] ** 2)

            m_hat = self.m[name] / (1.0 - beta1**step_idx)
            v_hat = self.v[name] / (1.0 - beta2**step_idx)

            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        self.W3 = params["W3"]
        self.b3 = params["b3"]

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def save_model(self, filepath="saved_models/own_system_mlp.npz", save_optimizer=False):
        """
        Zapisuje model do pliku .npz.

        Args:
            filepath (str): Ścieżka do pliku.
            save_optimizer (bool): Czy zapisywać stany Adama.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "input_dim": np.array([self.input_dim]),
            "hidden_dim": np.array([self.hidden_dim]),
            "output_dim": np.array([self.output_dim]),
            "seed": np.array([self.seed]),
            "best_epoch_nr": np.array([self.best_epoch_nr]),
        }

        if self.best_model_state is not None:
            save_dict["best_W1"] = self.best_model_state["W1"]
            save_dict["best_b1"] = self.best_model_state["b1"]
            save_dict["best_W2"] = self.best_model_state["W2"]
            save_dict["best_b2"] = self.best_model_state["b2"]
            save_dict["best_W3"] = self.best_model_state["W3"]
            save_dict["best_b3"] = self.best_model_state["b3"]

        if save_optimizer:
            save_dict["m_W1"] = self.m["W1"]
            save_dict["m_b1"] = self.m["b1"]
            save_dict["m_W2"] = self.m["W2"]
            save_dict["m_b2"] = self.m["b2"]
            save_dict["m_W3"] = self.m["W3"]
            save_dict["m_b3"] = self.m["b3"]

            save_dict["v_W1"] = self.v["W1"]
            save_dict["v_b1"] = self.v["b1"]
            save_dict["v_W2"] = self.v["W2"]
            save_dict["v_b2"] = self.v["b2"]
            save_dict["v_W3"] = self.v["W3"]
            save_dict["v_b3"] = self.v["b3"]

        np.savez(filepath, **save_dict)
        print(f"Model zapisany do: {filepath}")

    def load_model(self, filepath="saved_models/own_system_mlp.npz", load_optimizer=False):
        """
        Wczytuje model z pliku .npz.

        Args:
            filepath (str): Ścieżka do pliku.
            load_optimizer (bool): Czy wczytywać stany Adama.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {filepath}")

        data = np.load(filepath, allow_pickle=True)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]

        self.input_dim = int(data["input_dim"][0])
        self.hidden_dim = int(data["hidden_dim"][0])
        self.output_dim = int(data["output_dim"][0])
        self.seed = int(data["seed"][0])
        self.best_epoch_nr = int(data["best_epoch_nr"][0])

        if all(key in data.files for key in ["best_W1", "best_b1", "best_W2", "best_b2", "best_W3", "best_b3"]):
            self.best_model_state = {
                "W1": data["best_W1"],
                "b1": data["best_b1"],
                "W2": data["best_W2"],
                "b2": data["best_b2"],
                "W3": data["best_W3"],
                "b3": data["best_b3"],
            }
        else:
            self.best_model_state = None

        self._init_optimizer_states()

        if load_optimizer:
            opt_keys = [
                "m_W1", "m_b1", "m_W2", "m_b2", "m_W3", "m_b3",
                "v_W1", "v_b1", "v_W2", "v_b2", "v_W3", "v_b3",
            ]
            if all(key in data.files for key in opt_keys):
                self.m["W1"] = data["m_W1"]
                self.m["b1"] = data["m_b1"]
                self.m["W2"] = data["m_W2"]
                self.m["b2"] = data["m_b2"]
                self.m["W3"] = data["m_W3"]
                self.m["b3"] = data["m_b3"]

                self.v["W1"] = data["v_W1"]
                self.v["b1"] = data["v_b1"]
                self.v["W2"] = data["v_W2"]
                self.v["b2"] = data["v_b2"]
                self.v["W3"] = data["v_W3"]
                self.v["b3"] = data["v_b3"]

        print(f"Model wczytany z: {filepath}")

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        lr=0.001,
        epochs=100,
        patience=10,
        save_best_path=None,
        verbose=True,
    ):
        """
        Przeprowadza proces uczenia z walidacją, Early Stopping i zapisem najlepszego modelu.

        Args:
            X_train (np.ndarray): Dane treningowe [Trajektorie x Punkty x Cechy].
            y_train (np.ndarray): Cele treningowe [Trajektorie x Punkty x Wyjścia].
            X_val (np.ndarray): Dane walidacyjne.
            y_val (np.ndarray): Cele walidacyjne.
            lr (float): Learning rate.
            epochs (int): Maksymalna liczba epok.
            patience (int): Early stopping.
            save_best_path (str|None): Jeśli podane, zapisuje najlepszy model do pliku.
            verbose (bool): Czy wypisywać logi.
        """
        if verbose:
            print(f"\nStart treningu: {epochs} epok.")

        # Walidację spłaszczamy do 2D, bo sieć dostaje [N x cechy]
        if X_val.ndim == 3:
            X_val_eval = X_val.reshape(-1, X_val.shape[-1])
            y_val_eval = y_val.reshape(-1, y_val.shape[-1])
        else:
            X_val_eval = X_val
            y_val_eval = y_val

        best_val_loss = float("inf")
        epochs_no_improve = 0
        step_idx = 0

        epoch_bar = tqdm(range(epochs), desc="Trening epok", unit="epoka")

        for epoch in epoch_bar:
            train_losses = []

            traj_bar = tqdm(
                range(X_train.shape[0]),
                desc=f"Epoka {epoch + 1}/{epochs}",
                unit="traj",
                leave=False,
            )

            for i in traj_bar:
                x_traj = X_train[i]
                y_traj = y_train[i]

                y_pred, cache = self.forward(x_traj)
                loss = self.mse_loss(y_pred, y_traj)
                grads = self.backward(y_traj, cache)

                step_idx += 1
                self.adam_step(grads, lr=lr, step_idx=step_idx)

                train_losses.append(loss)
                traj_bar.set_postfix(loss=f"{loss:.6f}")

            avg_train_loss = float(np.mean(train_losses))

            # Walidacja
            v_pred = self.predict(X_val_eval)
            v_loss = float(self.mse_loss(v_pred, y_val_eval))

            # Early stopping + checkpoint
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(self.get_parameters())
                self.best_epoch_nr = epoch

                if save_best_path is not None:
                    self.save_model(save_best_path)
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix(
                train_loss=f"{avg_train_loss:.8f}",
                val_loss=f"{v_loss:.8f}",
                patience=f"{epochs_no_improve}/{patience}",
            )

            if verbose:
                print(
                    f"Epoka {epoch + 1:4d}/{epochs} | "
                    f"T_Loss: {avg_train_loss:.8f} | "
                    f"V_Loss: {v_loss:.8f} | "
                    f"Patience: {epochs_no_improve}/{patience}"
                )

            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"\nEarly Stopping! Brak poprawy przez {patience} epok. "
                        f"Aktualna epoka: {epoch + 1}"
                    )
                break

        # Przywrócenie najlepszego modelu
        if self.best_model_state is not None:
            self.set_parameters(self.best_model_state)
            if verbose:
                print(
                    f"Przywrócono najlepszy model "
                    f"(Val MSE: {best_val_loss:.8f}) z epoki {self.best_epoch_nr + 1}"
                )

            if save_best_path is not None:
                self.save_model(save_best_path)

    def simulate(self, t, u_new, h0, dh_dt0=None):
        """
        Symulacja rekurencyjna (Open-Loop) systemu przy użyciu nauczonego modelu.

        Args:
            t (np.ndarray): Wektor czasu [N].
            u_new (np.ndarray): Sygnał sterujący [N x dim_u] lub [N].
            h0 (np.ndarray): Warunek początkowy stanów [dim_y].
            dh_dt0 (np.ndarray, optional): Początkowa pochodna.

        Returns:
            SystemData: Obiekt z wynikami symulacji.
        """
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
                    u_new[i],      # u(k)
                    h_sim[i - 1],  # y(k-1)
                    dh_dt_prev,    # dy/dt(k-1)
                ]
            ).astype(float).reshape(1, -1)

            dh_dt_curr = self.predict(x_input).flatten()
            dh_dt_sim[i] = dh_dt_curr

            # Euler
            h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt

            # Ograniczenie fizyczne
            h_sim[i] = np.maximum(h_sim[i], 0.0)

            dh_dt_prev = dh_dt_curr

        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)