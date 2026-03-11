import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from SystemData import SystemData


class SystemMLP:
    """
    Klasa trenująca MLP na danych z SystemData i symulująca trajektorie.
    """
    def __init__(self, input_dim=2, hidden_dim=16):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def train(self, data: SystemData, lr=0.01, epochs=3000, print_every=500):
        X, y_target = data.get_training_data()
        X_tensor = self.torch.tensor(X, dtype=self.torch.float32)
        y_tensor = self.torch.tensor(y_target, dtype=self.torch.float32)

        criterion = self.nn.MSELoss()
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    def simulate(self, u_new, y0, dt=0.01):
        """
        Symuluje trajektorię y(t) dla nowego wymuszenia u_new.
        Obsługuje u_new jako wektor 1D lub macierz 2D.
        """
        n_points = len(u_new)
        y_sim = np.zeros(n_points)
        y_sim[0] = y0

        with self.torch.no_grad():
            for i in range(1, n_points):
                # spłaszczamy wejście
                u_flat = np.ravel(u_new[i - 1])
                # scalamy y i u w 1D array
                x_input = np.hstack(([y_sim[i - 1]], u_flat))
                # konwertujemy do np.array od razu
                x_input_np = np.array(x_input, dtype=np.float32)
                # tworzymy tensor 2D (1, input_dim)
                x_tensor = self.torch.tensor(x_input_np).unsqueeze(0)
                dy_dt_pred = self.model(x_tensor).item()
                y_sim[i] = y_sim[i - 1] + dy_dt_pred * dt
        return y_sim

