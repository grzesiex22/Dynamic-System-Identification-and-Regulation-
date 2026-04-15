import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy


class SystemMLP:
    """
        Model sieci neuronowej MLP do identyfikacji dynamiki systemów nieliniowych.
        Model przewiduje pochodne stanów: dy/dt = f(u(k), y(k-1), dy/dt(k-1)).
    """
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2):
        """
            Inicjalizacja architektury sieci.

            Args:
                input_dim (int): Liczba cech wejściowych (domyślnie 5: u, h1, h2, dh1, dh2).
                hidden_dim (int): Liczba neuronów w warstwach ukrytych.
                output_dim (int): Liczba wyjść (pochodne stanów dh1/dt, dh2/dt).
        """
        # Używamy Tanh dla gładkości, ale trzeba pamiętać o skalowaniu danych przy dużych amplitudach!
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Pobranie ścieżki do folderu, w którym znajduje się ten skrypt (Dataset)
        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(self.CURRENT_DIR)

        # Metadane i historia
        self.best_model_state = None
        self.training_config = {}
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "total_epochs": 0
        }

    def train(self, X_train, y_train, X_val, y_val, lr=0.001, epochs=100, patience=10):
        """
        Przeprowadza proces uczenia z walidacją, Early Stopping i zapisem najlepszego modelu.

        Args:
            X_train (np.ndarray): Dane treningowe [Trajektorie x Punkty x Cechy].
            y_train (np.ndarray): Cele treningowe (pochodne) [Trajektorie x Punkty x Wyjścia].
            X_val (np.ndarray): Dane walidacyjne.
            y_val (np.ndarray): Cele walidacyjne.
            lr (float): Współczynnik uczenia (Learning Rate).
            epochs (int): Maksymalna liczba epok.
            patience (int): Liczba epok bez poprawy przed zatrzymaniem (Early Stopping).
        """
        print(f"\nTORCH MLP - Start treningu: {epochs} epok, {X_train.shape[0]} trajektorii na epokę.")
        # Inicjalizacja konfiguracji
        self.training_config = {
            "lr": lr,
            "max_epochs": epochs,
            "patience": patience,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "optimizer": "Adam"
        }
        self.training_history["train_loss"] = []
        self.training_history["val_loss"] = []
        self.training_history["lr"] = []

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Konwersja walidacji na tensory raz, aby oszczędzić czas
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Inicjalizacja paska tqdm
        epoch_bar = tqdm(range(epochs), desc="Trening MLP", unit="epoka")

        for epoch in epoch_bar:
            self.model.train()
            train_losses = []

            # Trening po trajektoriach
            for i in range(X_train.shape[0]):
                x_traj = torch.tensor(X_train[i], dtype=torch.float32)
                y_traj = torch.tensor(y_train[i], dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = self.model(x_traj)
                loss = criterion(y_pred, y_traj)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # --- Sekcja Walidacji ---
            self.model.eval()
            with torch.no_grad():
                v_pred = self.model(X_val_t)
                v_loss = criterion(v_pred, y_val_t).item()

            # Zapis do historii
            self.training_history["train_loss"].append(float(avg_train_loss))
            self.training_history["val_loss"].append(float(v_loss))
            self.training_history["lr"].append(lr)
            self.training_history["total_epochs"] = epoch + 1

            # --- Logika Early Stopping i Checkpointing ---
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
                # Zapisujemy kopię najlepszych wag
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

        # Przywracamy najlepsze wagi znalezione podczas całego procesu
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            best_epoch = self.training_history["best_epoch"]
            print(f"✅ Przywrócono najlepszy model (Val MSE: {best_val_loss:.8f}) z epoki {best_epoch}")

    def simulate(self, t, u_new, h0, dh_dt0=None):
        """
            Symulacja rekurencyjna (Open-Loop) systemu przy użyciu nauczonego modelu.

            Args:
                t (np.ndarray): Wektor czasu [N].
                u_new (np.ndarray): Sygnał sterujący [N x dim_u].
                h0 (np.ndarray): Warunek początkowy stanów [dim_y].
                dh_dt0 (np.ndarray, optional): Początkowy trend pochodnych. Jeśli None, przyjmuje 0.

            Returns:
                SystemData: Obiekt zawierający wyniki pełnej symulacji.
        """

        n_points = len(t)
        dt = t[1] - t[0]

        # Inicjalizacja buforów na wyniki
        h_sim = np.zeros((n_points, self.output_dim))
        dh_dt_sim = np.zeros((n_points, self.output_dim))

        # Ustawienie punktu startowego
        h_sim[0] = h0
        dh_dt_prev = dh_dt0 if dh_dt0 is not None else np.zeros(self.output_dim)
        dh_dt_sim[0] = dh_dt_prev

        self.model.eval()
        with torch.no_grad():
            # Zabezpieczenie wymiarów sterowania
            if u_new.ndim == 1:
                u_new = u_new.reshape(-1, 1)

            for i in range(1, n_points):
                # 1. Przygotowanie wejścia modelu zgodnego z treningiem (u(k), h(k-1), dh/dt(k-1))
                x_input = np.concatenate([
                    u_new[i],       # To jest u(k)
                    h_sim[i - 1],   # To jest y(k-1)
                    dh_dt_prev      # To jest dy/dt(k-1)
                ]).astype(np.float32)

                # 2. Predykcja aktualnej pochodnej dh/dt(k)
                x_tensor = torch.tensor(x_input).unsqueeze(0)
                dh_dt_curr = self.model(x_tensor).numpy().flatten()
                dh_dt_sim[i] = dh_dt_curr

                # Całkowanie (Euler): h(k) = h(k-1) + dh_dt(k) * dt
                h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt

                # Ograniczenie fizyczne (brak ujemnych poziomów)
                h_sim[i] = np.maximum(h_sim[i], 0)

                # Aktualizacja trendu do następnego kroku
                dh_dt_prev = dh_dt_curr

        # ZWRACAMY GOTOWY OBIEKT SYSTEM DATA
        from Generators.SystemData import SystemData
        return SystemData(y=h_sim, u=u_new, t=t, dydt=dh_dt_sim)

    def save_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        """
        Zapisuje model do pliku .npz.
        Zapisuje metadane modelu do pliku .json.

        Args:
            folder (str): Ścieżka do podfolderu z modelami.
            dataset (str): nazwa datasetu
            base_name (str): nazwa modelu
        """

        full_dir = os.path.join(self.BASE_DIR, folder, dataset)
        os.makedirs(full_dir, exist_ok=True)

        base_path = os.path.join(full_dir, base_name)

        # 1. Zapis wag PyTorch
        torch.save(self.model.state_dict(), f"{base_path}.pth")

        # 2. Zapis metadanych do JSON
        metadata = {
            "model_arch": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            },
            "training_config": self.training_config,
            "training_history": self.training_history
        }

        with open(f"{base_path}_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"✅ Model i metadane zapisane w: {full_dir} jako {base_name}")

    def load_model(self, folder="Saved_models", dataset="Dataset1", base_name="Model"):
        """
        Wczytuje model z odpowiedniego podfolderu datasetu.
        """
        # Budujemy identyczną ścieżkę jak przy zapisie
        base_path = os.path.join(self.BASE_DIR, folder, dataset, base_name)
        metadata_path = f"{base_path}_info.json"
        model_path = f"{base_path}.pth"

        try:

            # 1. Wczytaj JSON, żeby wiedzieć jak zbudować sieć
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.input_dim = metadata["model_arch"]["input_dim"]
            self.hidden_dim = metadata["model_arch"]["hidden_dim"]
            self.output_dim = metadata["model_arch"]["output_dim"]
            self.training_config = metadata["training_config"]
            self.training_history = metadata["training_history"]

            # 2. Przebuduj model nn.Sequential (na wypadek gdyby parametry się zmieniły)
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )

            # 3. Wczytaj wagi binarne
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            print(f"📖 Model {base_name} wczytany z folderu {dataset}.")
            print(f"   Najlepsza epoka: {self.training_history.get('best_epoch', 'N/A')}")

        except FileNotFoundError:
                print(f"❌ BŁĄD: Nie znaleziono plików modelu {base_path} w {os.path.join(folder, dataset)}")
