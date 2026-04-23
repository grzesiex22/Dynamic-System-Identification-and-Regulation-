import json
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LSTMRegressor(nn.Module):
    """
    Prosty model LSTM:
    wejście:  [batch, seq_len, input_dim]
    wyjście:  [batch, output_dim]  -> predykcja dla ostatniego kroku sekwencji
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        y = self.fc(last_out)
        return y


class TorchLSTMSystem:
    """
    Model LSTM do identyfikacji dynamiki systemów nieliniowych.
    Model przewiduje pochodne stanów:
        dy/dt(k) = f( sekwencja [u, y, dy/dt] z ostatnich kroków )

    Uwaga:
    - do treningu budowane są okna sekwencyjne długości seq_len,
    - do symulacji model utrzymuje bufor ostatnich wejść.
    """

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, seq_len=10, num_layers=1, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )

        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(self.CURRENT_DIR)

        self.best_model_state = None
        self.training_config = {}
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "total_epochs": 0
        }

    @staticmethod
    def _create_sequences(X, Y, seq_len):
        """
        X: [n_traj, T, input_dim]
        Y: [n_traj, T, output_dim]

        Zwraca:
        X_seq: [n_samples, seq_len, input_dim]
        Y_seq: [n_samples, output_dim]
        """
        X_seq = []
        Y_seq = []

        for traj_idx in range(X.shape[0]):
            x_traj = X[traj_idx]
            y_traj = Y[traj_idx]

            if x_traj.shape[0] <= seq_len:
                continue

            for t in range(seq_len, x_traj.shape[0]):
                X_seq.append(x_traj[t - seq_len:t])
                Y_seq.append(y_traj[t])

        if len(X_seq) == 0:
            return (
                np.empty((0, seq_len, X.shape[-1]), dtype=np.float32),
                np.empty((0, Y.shape[-1]), dtype=np.float32)
            )

        return np.asarray(X_seq, dtype=np.float32), np.asarray(Y_seq, dtype=np.float32)

    def predict(self, X_seq):
        """
        X_seq: [batch, seq_len, input_dim]
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_seq, dtype=torch.float32)
            y = self.model(x_t).cpu().numpy()
        return y

    def train(self, X_train, y_train, X_val, y_val, lr=0.001, epochs=100, patience=10):
        """
        Trening LSTM na oknach sekwencyjnych zbudowanych z trajektorii.
        """
        print(f"\nTORCH LSTM - Start treningu: {epochs} epok, seq_len={self.seq_len}")

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, self.seq_len)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, self.seq_len)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise ValueError(
                f"Brak wystarczająco długich trajektorii dla seq_len={self.seq_len}. "
                f"Minimalna długość trajektorii musi być > seq_len."
            )

        self.training_config = {
            "lr": lr,
            "max_epochs": epochs,
            "patience": patience,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "optimizer": "Adam"
        }

        self.training_history["train_loss"] = []
        self.training_history["val_loss"] = []
        self.training_history["lr"] = []
        self.training_history["best_epoch"] = 0
        self.training_history["total_epochs"] = 0

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_seq, dtype=torch.float32)

        best_val_loss = float("inf")
        epochs_no_improve = 0

        epoch_bar = tqdm(range(epochs), desc="Trening LSTM", unit="epoka")

        for epoch in epoch_bar:
            self.model.train()
            train_losses = []

            for i in range(X_train_seq.shape[0]):
                x_sample = torch.tensor(X_train_seq[i:i+1], dtype=torch.float32)
                y_sample = torch.tensor(y_train_seq[i:i+1], dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = self.model(x_sample)
                loss = criterion(y_pred, y_sample)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            self.model.eval()
            with torch.no_grad():
                v_pred = self.model(X_val_t)
                v_loss = float(criterion(v_pred, y_val_t).item())

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(v_loss)
            self.training_history["lr"].append(lr)
            self.training_history["total_epochs"] = epoch + 1

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.training_history["best_epoch"] = epoch + 1
            else:
                epochs_no_improve += 1

            epoch_bar.set_postfix({
                "T_Loss": f"{avg_train_loss:.8f}",
                "V_Loss": f"{v_loss:.8f}",
                "Patience": f"{epochs_no_improve}/{patience}"
            })

            if epochs_no_improve >= patience:
                print(f"🛑 Early Stopping! Brak poprawy przez {patience} epok. Aktualna epoka: {epoch}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            best_epoch = self.training_history["best_epoch"]
            print(f"✅ Przywrócono najlepszy model (Val MSE: {best_val_loss:.8f}) z epoki {best_epoch}")

    def simulate(self, t, u_new, h0, dh_dt0=None):
        """
        Symulacja rekurencyjna LSTM.

        Bufor wejściowy ma długość seq_len i na każdym kroku zawiera wektory:
            [u(k), y(k-1), dy/dt(k-1)]

        Na początku bufor jest wypełniany powtarzanym wektorem startowym.
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

        initial_feature = np.concatenate([
            u_new[0],
            h_sim[0],
            dh_dt_prev
        ]).astype(np.float32)

        buffer = np.tile(initial_feature, (self.seq_len, 1)).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(1, n_points):
                current_feature = np.concatenate([
                    u_new[i],
                    h_sim[i - 1],
                    dh_dt_prev
                ]).astype(np.float32)

                buffer[:-1] = buffer[1:]
                buffer[-1] = current_feature

                x_seq = torch.tensor(buffer[np.newaxis, :, :], dtype=torch.float32)
                dh_dt_curr = self.model(x_seq).cpu().numpy().flatten().astype(np.float32)

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

        torch.save(self.model.state_dict(), f"{base_path}.pt")

        metadata = {
            "model_arch": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seq_len": self.seq_len,
                "num_layers": self.num_layers,
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
        model_path = f"{base_path}.pt"

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.hidden_dim = metadata["model_arch"]["hidden_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.seq_len = metadata["model_arch"]["seq_len"]
            self.num_layers = metadata["model_arch"]["num_layers"]
            self.seed = metadata["model_arch"]["seed"]

            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            self.model = LSTMRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers
            )

            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
            print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")