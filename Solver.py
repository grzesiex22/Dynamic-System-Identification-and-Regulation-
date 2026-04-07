from scipy.integrate import solve_ivp
from Objects.DynamicObject import DynamicObject


class Solver:
    """
    Uniwersalny silnik obliczeniowy do całkowania równań różniczkowych zwyczajnych (ODE).

    Klasa stanowi interfejs pomiędzy zdefiniowanymi modelami fizycznymi (systemami)
    a zaawansowanym solverem numerycznym RK45. Pozwala na uzyskanie dokładnych
    trajektorii stanów systemu dla stałych parametrów wymuszenia.
    """

    def __init__(self, system: DynamicObject):
        """
        Inicjalizuje solver dla konkretnego systemu dynamicznego.

        Args:
            system (DynamicObject): Instancja klasy dziedziczącej po DynamicObject,
                zawierająca definicję funkcji f, zakres czasu t_span oraz
                warunki początkowe x0.
        """
        self.system = system

    def solve(self):
        """
        Wykonuje numeryczne rozwiązanie zagadnienia początkowego (Cauchy'ego).

        Metoda wykorzystuje algorytm Rungego-Kutty rzędu 4(5) z adaptacyjnym
        krokiem czasowym, co zapewnia wysoką precyzję obliczeń przy zachowaniu
        stabilności numerycznej.

        Returns:
            OdeResult: Obiekt biblioteki scipy zawierający:
                - t (np.ndarray): Węzły czasowe wybrane przez solver.
                - y (np.ndarray): Wartości stanów w odpowiadających chwilach czasu.
                - success (bool): Informacja, czy całkowanie zakończyło się pomyślnie.
        """
        # solve_ivp automatycznie dobiera krok całkowania, aby zminimalizować błąd
        sol = solve_ivp(
            self.system.f,  # Funkcja dx/dt = f(t, x, *args)
            self.system.t_span,  # Przedział czasu (t_start, t_end)
            self.system.x0,  # Wektor stanu początkowego
            method="RK45",  # Jawna metoda Rungego-Kutty rzędu 5(4)
            args=self.system.params  # Dodatkowe parametry przekazywane do funkcji f
        )
        return sol
