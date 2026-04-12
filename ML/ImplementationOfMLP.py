import numpy as np
import copy

class OwnSystemMLP:
    """
    Model sieci neuronowej MLP do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt = f(u(k), y(k-1), dy/dt(k-1))
    """

   def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, seed=42):
    """
    Inicjalizacja architektury sieci.

    Args:
        input_dim (int): Liczba cech wejściowych.
        hidden_dim (int): Liczba neuronów w warstwach ukrytych.
        output_dim (int): Liczba wyjść.
        seed (int): Ziarno losowości.
    """
    rng = np.random.default_rng(seed)

    self.W1 = rng.normal(0.0, 1.0, size=(input_dim, hidden_dim))
    self.b1 = np.zeros((1, hidden_dim))

    self.W2 = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim))
    self.b2 = np.zeros((1, hidden_dim))

    self.W3 = rng.normal(0.0, 1.0, size=(hidden_dim, output_dim))
    self.b3 = np.zeros((1, output_dim))

    self.output_dim = output_dim
    self.best_model_state = None
    self.best_epoch_nr = 0

    # Stany optimizera Adam
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
        return 1.0 - a ** 2

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
            "y_pred": y_pred
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
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3
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
            "b3": self.b3
        }

        for name in params:
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grads[name] ** 2)

            m_hat = self.m[name] / (1.0 - beta1 ** step_idx)
            v_hat = self.v[name] / (1.0 - beta2 ** step_idx)

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

    def train(self, X_train, y_train, X_val, y_val, lr=0.001, epochs=100, patience=10):
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
        """
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

        for epoch in range(epochs):
            train_losses = []

            # trening po trajektoriach, tak jak w wersji torch
            for i in range(X_train.shape[0]):
                x_traj = X_train[i]
                y_traj = y_train[i]

                y_pred, cache = self.forward(x_traj)
                loss = self.mse_loss(y_pred, y_traj)
                grads = self.backward(y_traj, cache)

                step_idx += 1
                self.adam_step(grads, lr=lr, step_idx=step_idx)

                train_losses.append(loss)

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
            else:
                epochs_no_improve += 1

            print(
                f"Epoka {epoch + 1:4d}/{epochs} | "
                f"T_Loss: {avg_train_loss:.8f} | "
                f"V_Loss: {v_loss:.8f} | "
                f"Patience: {epochs_no_improve}/{patience}"
            )

            if epochs_no_improve >= patience:
                print(f"\nEarly Stopping! Brak poprawy przez {patience} epok. Aktualna epoka: {epoch}")
                break

        # Przywrócenie najlepszego modelu
        if self.best_model_state is not None:
            self.set_parameters(self.best_model_state)
            print(f"Przywrócono najlepszy model (Val MSE: {best_val_loss:.8f}) z epoki {self.best_epoch_nr}")

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
            x_input = np.concatenate([
                u_new[i],        # u(k)
                h_sim[i - 1],    # y(k-1)
                dh_dt_prev       # dy/dt(k-1)
            ]).astype(float).reshape(1, -1)

            dh_dt_curr = self.predict(x_input).flatten()
            dh_dt_sim[i] = dh_dt_curr

            # Euler
            h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt

            # Ograniczenie fizyczne
            h_sim[i] = np.maximum(h_sim[i], 0.0)

            dh_dt_prev = dh_dt_curr

        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)