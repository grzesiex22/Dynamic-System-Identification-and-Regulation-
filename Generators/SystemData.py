import numpy as np


class SystemData:
    """
    Kontener danych procesowych systemu dynamicznego i generator cech (Feature Engineering).

    Klasa przechowuje surowe wyniki symulacji oraz odpowiada za ich transformację
    do formatu wektorowego (X, y) zgodnego z przyczynowością systemów dynamicznych.
    Implementuje przesunięcia czasowe (lagging) niezbędne do predykcji pochodnych.
    """

    def __init__(self, y, u, t, dydt=None):
        """
        Inicjalizuje obiekt i wyznacza bazowe pochodne sygnałów.

        Args:
            y (np.ndarray): Macierz stanów wyjściowych (np. poziomy cieczy) [N x dim_y].
            u (np.ndarray): Macierz sygnałów wymuszających (np. napięcia) [N x dim_u].
            t (np.ndarray): Wektor czasu [N].
            dydt (np.ndarray, optional): Gotowe pochodne. Jeśli brak, zostaną wyliczone.

        Attributes:
            dt (float): Krok próbkowania (wyznaczony z t).
            dy_dt (np.ndarray): Pochodne stanów. Jeśli liczone automatycznie, mają wymiar [N-1 x dim_y].
        """
        self.y = np.array(y)
        self.u = np.array(u)
        self.t = np.array(t)
        self.dt = t[1] - t[0] if len(t) > 1 else 0.0

        # Wyznaczanie pochodnych różnicą w przód (Forward Difference)
        if dydt is None:
            self.dy_dt = np.diff(self.y, axis=0) / self.dt
        else:
            self.dy_dt = dydt

    def get_training_data(self):
        """
        Przygotowuje zbiór uczący (X, y_target) dla modelu autoregresyjnego.

        Zapewnia synchronizację sygnałów zgodnie z modelem:
        dy/dt(k) = f( u(k), y(k-1), dy/dt(k-1) )

        Dzięki temu model uczy się dynamiki (zmiany), a nie tylko statycznej mapy.
        Operacja skraca dane o 2 punkty (jeden przez np.diff, drugi przez przesunięcie).

        Returns:
            tuple: (X, y_target) gdzie:
                X (np.ndarray): [u(k), y(k-1), dy/dt(k-1)] o wymiarze [N-2 x 5].
                y_target (np.ndarray): [dy/dt(k)] o wymiarze [N-2 x 2].
        """

        # Cel (TARGET): Aktualna zmiana systemu
        # Skracamy do N-2, aby móc spojrzeć wstecz dla każdego punktu
        y_target = self.dy_dt[1:]

        # Wejście (INPUT) u(k): Sterowanie przyłożone w danej chwili
        u_current = self.u[1:len(y_target) + 1]
        if u_current.ndim == 1:
            u_current = u_current.reshape(-1, 1)

        # Wejście (INPUT) y(k-1): Stan systemu tuż przed zmianą
        y_prev = self.y[:len(y_target)]

        # Wejście (INPUT) dy/dt(k-1): Poprzedni trend (bezwładność)
        dy_dt_prev = self.dy_dt[:len(y_target)]

        # Połączenie cech w jedną macierz wejściową
        X = np.hstack((u_current, y_prev, dy_dt_prev))

        return X, y_target

    def get_data_to_simulate(self):
        """
        Zwraca kompletne sygnały sterujące i warunki początkowe do symulacji Open-Loop.

        Metoda przygotowuje surowe dane tak, aby symulacja mogła wystartować
        z tego samego punktu, co dane rzeczywiste.

        Returns:
            tuple: (t_sim, u_to_sim, y0, dy_dt0)
                t_sim: Pełny wektor czasu.
                u_to_sim: Macierz wymuszeń [N x dim_u].
                y0: Wektor stanu początkowego [dim_y].
                dy_dt0: Wektor trendu początkowego [dim_y].
        """

        t_sim = self.t
        y0 = self.y[0]
        dy_dt0 = self.dy_dt[0]  # Pierwsza dostępna pochodna

        if self.u.ndim == 1:
            u_to_sim = self.u.reshape(-1, 1)
        else:
            u_to_sim = self.u

        return t_sim, u_to_sim, y0, dy_dt0

    def get_data_to_plot(self):
        """
        Wyrównuje wszystkie sygnały do wspólnego wymiaru (N-2) na potrzeby wizualizacji.

        Gwarantuje, że na wykresie punkt w czasie 't' odpowiada dokładnie
        wartościom u, y oraz pochodnej, które zostały użyte w treningu.
        Eliminuje to przesunięcia fazowe na wykresach.

        Returns:
            tuple: (t_plot, u_plot, y_plot, dydt_plot)
                Wszystkie tablice mają wymiar [N-2].
        """

        # Obliczamy długość docelową (ilość dostępnych par wejście-wyjście)
        n_target = len(self.t) - 2

        # Wycinamy dane omijając pierwszy punkt (brak trendu wstecznego) i ostatni (brak pochodnej docelowej)
        t_plot = self.t[1:1 + n_target]
        y_plot = self.y[1:1 + n_target, :]

        # Sterowanie u
        if self.u.ndim == 1:
            u_plot = self.u[1:1 + n_target].reshape(-1, 1)
        else:
            u_plot = self.u[1:1 + n_target, :]

        # Pochodne (aktualne dy/dt(k))
        dydt_plot = self.dy_dt[1:1 + n_target, :]

        return t_plot, u_plot, y_plot, dydt_plot