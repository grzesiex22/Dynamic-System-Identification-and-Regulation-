import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class SystemMLP:
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim

    def train(self, X_train, y_train, X_val, y_val, lr=0.001, epochs=100):
        # X_train: (1000, 1998, 5)
        # y_train: (1000, 1998, 2)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        # Inicjalizacja paska tqdm
        pbar = tqdm(range(epochs), desc="Trening MLP")

        print(f"Start treningu: {epochs} epok, 1000 trajektorii na epokę.")

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0

            # Pętla po KAŻDEJ trajektorii osobno
            for i in range(X_train.shape[0]):
                # Wyciągamy jedną całą trajektorię (1998, 5)
                x_traj = torch.tensor(X_train[i], dtype=torch.float32)
                y_traj = torch.tensor(y_train[i], dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = self.model(x_traj)
                loss = criterion(y_pred, y_traj)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Obliczanie średniego błędu
            avg_train_loss = epoch_loss / X_train.shape[0]

            # Walidacja raz na epokę na całym zbiorze walidacyjnym
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Walidacja też trajektoria po trajektorii dla czystości
                    v_pred = self.model(X_val_t)
                    v_loss = criterion(v_pred, y_val_t)

                avg_loss = epoch_loss / X_train.shape[0]
                print(f"Epoka {epoch} | Średni Train MSE: {avg_loss:.8f} | Val MSE: {v_loss.item():.8f}")

            # AKTUALIZACJA PASKA: dodajemy info o błędzie na końcu paska
            pbar.set_postfix({
                "Loss": f"{avg_train_loss:.6f}",
                "Val": f"{v_loss.item():.6f}"
            })

    def simulate(self, u_new, h0, dt):
        """
        Symulacja rekurencyjna systemu zbiorników.

        Args:
            u_new (np.ndarray): Sterowanie [N x 1]
            h0 (np.ndarray): Początkowe poziomy [h1, h2]
            dt (float): Krok próbkowania
        """
        n_points = len(u_new)
        h_sim = np.zeros((n_points, self.output_dim))
        dh_dt_sim = np.zeros((n_points, self.output_dim))

        h_sim[0] = h0
        dh_dt_prev = np.zeros(self.output_dim)

        self.model.eval()
        with torch.no_grad():
            for i in range(1, n_points):
                # Wejście: [u(k), h(k-1), dh_dt(k-1)]
                x_input = np.concatenate([
                    u_new[i],  # u(k)
                    h_sim[i - 1],  # h(k-1)
                    dh_dt_prev  # dh_dt(k-1)
                ]).astype(np.float32)

                x_tensor = torch.tensor(x_input).unsqueeze(0)

                # Model przewiduje AKTUALNĄ pochodną dh_dt(k)
                dh_dt_curr = self.model(x_tensor).numpy().flatten()
                dh_dt_sim[i] = dh_dt_curr

                # Całkowanie (Euler): h(k) = h(k-1) + dh_dt(k) * dt
                h_sim[i] = h_sim[i - 1] + dh_dt_curr * dt

                # Ograniczenie fizyczne (brak ujemnych poziomów)
                h_sim[i] = np.maximum(h_sim[i], 0)

                # Aktualizacja trendu do następnego kroku
                dh_dt_prev = dh_dt_curr

        return h_sim, dh_dt_sim
