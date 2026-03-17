import numpy as np


class SystemData:
    """
    Kontener danych procesowych systemu dynamicznego.

    Klasa przechowuje surowe wyniki symulacji (czas, stany, sterowania) oraz
    odpowiada za ich transformację do formatu macierzowego (X, y) wymaganego
    przez algorytmy uczenia maszynowego.
    """

    def __init__(self, y, u, t):
        """
        Inicjalizuje obiekt SystemData i oblicza pochodne sygnałów wyjściowych.

        Args:
            y (list/np.ndarray): Macierz stanów wyjściowych systemu [N x wymiar_stanu].
            u (list/np.ndarray): Wektor lub macierz sygnałów sterujących [N x wymiar_u].
            t (list/np.ndarray): Wektor czasu [N].

        Attributes:
            dt (float): Krok próbkowania wyliczony na podstawie wektora czasu.
            dy_dt (np.ndarray): Pochodne stanów obliczone metodą różnic centralnych (gradient).
        """
        self.y = np.array(y)
        self.u = np.array(u)
        self.t = np.array(t)
        self.dt = t[1] - t[0] if len(t) > 1 else 0.0
        # Obliczanie pochodnej dla celów wizualizacji (cała długość wektora)
        self.dy_dt = np.gradient(self.y, self.dt, axis=0)

    def get_training_data(self):
        """
        Przygotowuje zbiór uczący (X, y_target) do identyfikacji modelu dy/dt = f(u, y).

        Proces obejmuje synchronizację wymiarów (wyrównanie u(k) i y(k) do pochodnej)
        oraz obliczenie pochodnych w przód (forward difference), co jest standardem
        przy modelowaniu systemów dyskretnych.

        Returns:
            tuple: (X, y_target)
                - X: Macierz cech [N-1 x (wymiar_u + wymiar_y)]. Zawiera pary [u(k), y(k)].
                - y_target: Macierz celów [N-1 x wymiar_y]. Zawiera pochodne dy/dt(k).
        """
        # 1. Obliczamy pochodne w przód: (h(k+1) - h(k)) / dt
        # diff zmniejsza liczbę próbek o 1 (wymiar N-1)
        dy_dt_forward = np.diff(self.y, axis=0) / self.dt

        # 2. Synchronizacja sterowania u - ucinamy ostatni element
        u_cut = self.u[:-1]
        # Wymuszamy postać macierzy kolumnowej [N-1, 1], jeśli u jest wektorem 1D
        if u_cut.ndim == 1:
            u_cut = u_cut.reshape(-1, 1)

        # 3. Synchronizacja stanów y - ucinamy ostatni element
        y_cut = self.y[:-1]

        # 4. Agregacja danych wejściowych do sieci (Feature Engineering)
        # Łączymy kolumny sterowania i stanów: [u, y1, y2, ...]
        X = np.hstack((u_cut, y_cut))
        y_target = dy_dt_forward

        return X, y_target

    def get_data_to_plot(self):
        """
        Zwraca komplet danych w formacie czytelnym dla klasy SystemPlotter.

        Returns:
            tuple: (t, y, dy_dt, u) - pełne trajektorie czasowe.
        """
        return self.t, self.y, self.dy_dt, self.u