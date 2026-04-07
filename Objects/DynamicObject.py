import numpy as np


class DynamicObject:
    """
    Abstrakcyjna klasa bazowa reprezentująca obiekt dynamiczny sterowany sygnałem u.

    Służy jako szablon dla konkretnych modeli fizycznych. Każda klasa dziedzicząca
    musi zaimplementować równania różniczkowe w metodzie f.
    """

    def __init__(self):
        """
        Inicjalizuje podstawowe parametry wymagane przez solvery ODE.

        Attributes:
            x0 (list): Wektor stanu początkowego systemu (np. początkowe poziomy, prędkości).
            t_span (tuple): Zakres czasu symulacji (czas_start, czas_stop) w sekundach.
            params (dict): Słownik przechowujący parametry fizyczne (stałe, masy, wymiary).
        """
        self.x0 = [0.0, 0.0]
        self.t_span = (0, 100)
        self.params = {}

    def f(self, t, x, u):
        """
        Definiuje funkcję przejścia (równania stanu) systemu: dx/dt = f(t, x, u).

        Args:
            t (float): Aktualny czas symulacji.
            x (np.ndarray): Aktualny wektor stanu systemu.
            u (float/np.ndarray): Wartość sygnału sterującego w danej chwili t.

        Returns:
            list/np.ndarray: Wektor pochodnych stanów po czasie (dx/dt).

        Raises:
            NotImplementedError: Jeśli metoda nie zostanie nadpisana w klasie pochodnej.
        """
        raise NotImplementedError
