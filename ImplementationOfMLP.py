import numpy as np
import copy


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


class SystemMLP:
    def __init__(self, input_dim=3, hidden_layers=None, output_dim=2, seed=42):
        if hidden_layers is None:
            hidden_layers = [64, 64]

        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.output_dim = output_dim
        self.seed = seed

        np.random.seed(seed)

        layer_dims = [input_dim] + self.hidden_layers + [output_dim]

        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
            b = np.zeros((1, out_dim))

            self.weights.append(W)
            self.biases.append(b)

        self.best_weights = copy.deepcopy(self.weights)
        self.best_biases = copy.deepcopy(self.biases)
        self.best_val_loss = float("inf")

        self.train_loss_history = []
        self.val_loss_history = []

    def summary(self):
        print("=== SystemMLP (NumPy) ===")
        print(f"Wejścia       : {self.input_dim}")
        print(f"Warstwy ukryte: {self.hidden_layers}")
        print(f"Wyjścia       : {self.output_dim}")
        dims = [self.input_dim] + self.hidden_layers + [self.output_dim]
        print("Architektura  :", " -> ".join(map(str, dims)))

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_derivative(a):
        return 1.0 - a ** 2

    @staticmethod
    def _train_val_split(X, y, val_ratio=0.2, shuffle=True, seed=42):
        n = len(X)
        indices = np.arange(n)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        split = int(n * (1.0 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _forward(self, X):
        activations = [X]
        A = X

        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self._tanh(Z)
            activations.append(A)

        Z_out = A @ self.weights[-1] + self.biases[-1]
        y_pred = Z_out
        activations.append(y_pred)

        return activations, y_pred

    def _backward(self, activations, y_true, y_pred):
        n = y_true.shape[0]

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        dA = (2.0 / n) * (y_pred - y_true)

        A_prev = activations[-2]
        grad_weights[-1] = A_prev.T @ dA
        grad_biases[-1] = np.sum(dA, axis=0, keepdims=True)

        dA_prev = dA @ self.weights[-1].T

        for layer in range(len(self.weights) - 2, -1, -1):
            A_curr = activations[layer + 1]
            A_prev = activations[layer]

            dZ = dA_prev * self._tanh_derivative(A_curr)

            grad_weights[layer] = A_prev.T @ dZ
            grad_biases[layer] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA_prev = dZ @ self.weights[layer].T

        return grad_weights, grad_biases

    def train(
        self,
        X,
        y,
        epochs=1000,
        lr=0.001,
        val_ratio=0.2,
        shuffle=True,
        print_every=100
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X musi mieć wymiar [n_samples, n_features].")

        if y.ndim != 2:
            raise ValueError("y musi mieć wymiar [n_samples, n_outputs].")

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X ma {X.shape[1]} cech, a model oczekuje {self.input_dim}."
            )

        if y.shape[1] != self.output_dim:
            raise ValueError(
                f"y ma {y.shape[1]} wyjść, a model oczekuje {self.output_dim}."
            )

        X_train, y_train, X_val, y_val = self._train_val_split(
            X, y, val_ratio=val_ratio, shuffle=shuffle, seed=self.seed
        )

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")
        self.best_weights = copy.deepcopy(self.weights)
        self.best_biases = copy.deepcopy(self.biases)

        print(f"Start treningu: {epochs} epok")

        for epoch in range(epochs):
            activations, y_pred_train = self._forward(X_train)
            train_loss = Metrics.mse(y_train, y_pred_train)

            grad_weights, grad_biases = self._backward(activations, y_train, y_pred_train)

            for i in range(len(self.weights)):
                self.weights[i] -= lr * grad_weights[i]
                self.biases[i] -= lr * grad_biases[i]

            _, y_pred_val = self._forward(X_val)
            val_loss = Metrics.mse(y_val, y_pred_val)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(self.weights)
                self.best_biases = copy.deepcopy(self.biases)

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(
                    f"Epoka {epoch:5d} | "
                    f"train MSE: {train_loss:.8f} | "
                    f"val MSE: {val_loss:.8f}"
                )

        self.weights = copy.deepcopy(self.best_weights)
        self.biases = copy.deepcopy(self.best_biases)

        print(f"Trening zakończony. Najlepsze val MSE: {self.best_val_loss:.8f}")

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Niepoprawny wymiar wejścia. Oczekiwano {self.input_dim}, otrzymano {X.shape[1]}."
            )

        _, y_pred = self._forward(X)
        return y_pred

    def predict_on_system_data(self, data):
        X, y_true = data.get_training_data()
        y_pred = self.predict(X)
        return y_true, y_pred

    def simulate(self, u_new, y0, dt):
        u_new = np.asarray(u_new, dtype=np.float64)

        if u_new.ndim == 1:
            u_new = u_new.reshape(-1, 1)

        y0 = np.asarray(y0, dtype=np.float64).flatten()

        if len(y0) != self.output_dim:
            raise ValueError(
                f"Stan początkowy y0 ma długość {len(y0)}, a output_dim={self.output_dim}."
            )

        n_points = len(u_new)
        y_sim = np.zeros((n_points, self.output_dim), dtype=np.float64)
        y_sim[0] = y0

        for i in range(1, n_points):
            u_curr = u_new[i - 1]
            y_prev = y_sim[i - 1]

            x_input = np.concatenate([u_curr, y_prev]).reshape(1, -1)

            if x_input.shape[1] != self.input_dim:
                raise ValueError(
                    f"Podczas symulacji wejście ma wymiar {x_input.shape[1]}, "
                    f"a model oczekuje {self.input_dim}."
                )

            dy_dt_pred = self.predict(x_input).flatten()
            y_sim[i] = y_prev + dy_dt_pred * dt
            y_sim[i] = np.maximum(y_sim[i], 0.0)

        return y_sim

    def evaluate_derivatives(self, data):
        y_true, y_pred = self.predict_on_system_data(data)
        return Metrics.evaluate(y_true, y_pred)

    def evaluate_simulation(self, y_true, y_sim):
        return Metrics.evaluate(y_true, y_sim)

    def save_best_model(self, filepath):
        data = {}

        for i, W in enumerate(self.best_weights):
            data[f"W{i}"] = W

        for i, b in enumerate(self.best_biases):
            data[f"b{i}"] = b

        data["input_dim"] = np.array([self.input_dim])
        data["output_dim"] = np.array([self.output_dim])
        data["hidden_layers"] = np.array(self.hidden_layers)
        data["best_val_loss"] = np.array([self.best_val_loss])

        np.savez(filepath, **data)

    def load_model(self, filepath):
        data = np.load(filepath, allow_pickle=True)

        loaded_input_dim = int(data["input_dim"][0])
        loaded_output_dim = int(data["output_dim"][0])
        loaded_hidden_layers = list(data["hidden_layers"])

        if loaded_input_dim != self.input_dim:
            raise ValueError(
                f"input_dim z pliku = {loaded_input_dim}, a bieżący model ma {self.input_dim}"
            )

        if loaded_output_dim != self.output_dim:
            raise ValueError(
                f"output_dim z pliku = {loaded_output_dim}, a bieżący model ma {self.output_dim}"
            )

        if loaded_hidden_layers != self.hidden_layers:
            raise ValueError(
                f"hidden_layers z pliku = {loaded_hidden_layers}, a bieżący model ma {self.hidden_layers}"
            )

        self.weights = []
        self.biases = []

        n_layers = len([key for key in data.keys() if key.startswith("W")])

        for i in range(n_layers):
            self.weights.append(data[f"W{i}"])
            self.biases.append(data[f"b{i}"])

        self.best_weights = copy.deepcopy(self.weights)
        self.best_biases = copy.deepcopy(self.biases)
        self.best_val_loss = float(data["best_val_loss"][0])