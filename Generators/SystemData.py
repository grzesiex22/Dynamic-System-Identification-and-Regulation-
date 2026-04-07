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
        # Obliczanie pochodnej
        self.dy_dt = np.diff(self.y, axis=0) / self.dt
        self.y = self.y[:-1]
        self.u = self.u[:-1]
        self.t = self.t[:-1]

    def get_training_data(self):
        """
        Przygotowuje zbiór uczący (X, y_target) do identyfikacji modelu dy/dt = f(u, y).

        Proces obejmuje synchronizację wymiarów (wyrównanie u(k) i y(k) do pochodnej)
        oraz obliczenie pochodnych w przód (forward difference), co jest standardem
        przy modelowaniu systemów dyskretnych.

        Model: dy/dt(k) = f( u(k), y(k-1), dy/dt(k-1) )

        Returns:
            X: [u(k), h1(k-1), h2(k-1), dh1/dt(k-1), dh2/dt(k-1)] -> Macierz (N-2, 5)
            Y: [dh1/dt(k), dh2/dt(k)] -> Macierz (N-2, 2)
        """
        # TARGET: dh/dt(k) -> bierzemy od 1 do końca
        y_target = self.dy_dt[1:]

        # INPUT u: u(k) -> bierzemy od 1 do końca (bo u zostało już skrócone w init do N-1)
        u_current = self.u[1:]
        if u_current.ndim == 1:
            u_current = u_current.reshape(-1, 1)

        # INPUT y: h(k-1) -> bierzemy od początku do przedostatniego
        y_prev = self.y[:-1]

        # INPUT dy/dt: dh/dt(k-1) -> bierzemy od początku do przedostatniego
        dy_dt_prev = self.dy_dt[:-1]

        # Hstack połączy to w (N-2, 5)
        X = np.hstack((u_current, y_prev, dy_dt_prev))

        return X, y_target, self.t

    def get_data_to_plot(self):
        """
        Zwraca komplet danych w formacie czytelnym dla klasy SystemPlotter.

        Returns:
            tuple: (t, y, dy_dt, u) - pełne trajektorie czasowe.
        """
        return self.t, self.y, self.dy_dt, self.u