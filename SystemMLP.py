import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from SystemData import SystemData


class SystemMLP:
    """
    Model sieci neuronowej typu Multi-Layer Perceptron (MLP) do identyfikacji dynamiki systemów.

    Sieć pełni rolę uniwersalnego aproksymatora funkcji przejścia systemu. Uczy się
    wyznaczać pochodne stanów (dy/dt) na podstawie aktualnego wymuszenia (u)
    oraz obecnego stanu (y).
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        """
        Konstruuje architekturę sieci neuronowej.

        Args:
            input_dim (int): Wymiar wejścia. Dla układu zbiorników: u + h1 + h2 = 3.
            hidden_dim (int): Liczba neuronów w warstwach ukrytych.
            output_dim (int): Wymiar wyjścia (liczba przewidywanych pochodnych).

        Note:
            Zastosowano funkcję Tanh zamiast ReLU, ponieważ Tanh jest funkcją gładką
            i różniczkowalną w każdym punkcie, co lepiej oddaje ciągły charakter
            zjawisk fizycznych (jak przepływ cieczy).
        """
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim

    def train(self, data:SystemData, lr=0.001, epochs=3000, print_every=500):
        """
        Przeprowadza proces uczenia nadzorowanego (Supervised Learning).

        Args:
            data (SystemData): Obiekt z danymi treningowymi.
            lr (float): Współczynnik uczenia (Learning Rate).
            epochs (int): Liczba epok (iteracji optymalizatora).
            print_every (int): Częstotliwość raportowania błędu MSE w konsoli.

        Process:
            1. Pobranie danych zsynchronizowanych czasowo [u(k), y(k)] -> [dy/dt(k)].
            2. Minimalizacja błędu średniokwadratowego (MSE) za pomocą algorytmu Adam.
        """
        X, y_target = data.get_training_data()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"Rozpoczęcie treningu modelu: {epochs} epok...")
        for epoch in range(epochs):
            optimizer.zero_grad()  # Resetowanie gradientów
            y_pred = self.model(X_tensor)  # Propagacja w przód
            loss = criterion(y_pred, y_tensor)  # Obliczanie błędu
            loss.backward()  # Propagacja wsteczna (backpropagation)
            optimizer.step()  # Aktualizacja wag

            if epoch % print_every == 0:
                print(f"Epoka {epoch}, Strata (MSE): {loss.item():.8f}")

    def simulate(self, u_new, y0, dt):
        """
        Wykonuje predykcję wielokrokową (symulację rekurencyjną).

        Jest to kluczowy test modelu: sieć nie dostaje prawdziwych stanów y w każdym
        kroku, lecz bazuje na własnych, wcześniej przewidzianych wartościach.
        Pozwala to ocenić, czy model poprawnie "zrozumiał" dynamikę obiektu.

        Args:
            u_new (np.ndarray): Macierz wymuszeń testowych [N x 1].
            y0 (np.ndarray): Stan początkowy [h1, h2].
            dt (float): Krok integracji numerycznej (zgodny z obiektem).

        Returns:
            np.ndarray: Symulowana trajektoria stanów systemu [N x output_dim].
        """
        n_points = len(u_new)
        y_sim = np.zeros((n_points, self.output_dim))
        y_sim[0] = y0

        self.model.eval()  # Wyłączenie mechanizmów treningowych (np. Dropout)
        with torch.no_grad():  # Wyłączenie śledzenia gradientów (oszczędność pamięci i czasu)
            for i in range(1, n_points):
                # 1. Budowa wektora wejściowego z obecnego sterowania i poprzedniej predykcji
                u_curr = u_new[i - 1]
                y_prev = y_sim[i - 1]

                # 2. Przygotowanie tensora wejściowego [batch_size=1, input_dim]
                x_input = np.concatenate([u_curr, y_prev]).astype(np.float32)
                x_tensor = torch.tensor(x_input).unsqueeze(0)

                # 3. Predykcja pochodnych (dy/dt) przez sieć neuronową
                dy_dt_pred = self.model(x_tensor).numpy().flatten()

                # 4. Integracja metodą Eulera (wyznaczenie stanu w następnej chwili czasu)
                # y(k) = y(k-1) + f_NN(y(k-1), u(k-1)) * dt
                y_sim[i] = y_sim[i - 1] + dy_dt_pred * dt

                # 5. Zabezpieczenie przed wartościami nierealnymi fizycznie (h < 0)
                y_sim[i] = np.maximum(y_sim[i], 0)

        return y_sim
