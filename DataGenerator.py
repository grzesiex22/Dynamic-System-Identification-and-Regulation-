import numpy as np
from scipy.integrate import solve_ivp
from SystemData import SystemData
from DynamicObject import DynamicObject


class DataGenerator:
    """
    Klasa odpowiedzialna za symulację numeryczną obiektów dynamicznych.

    Wykorzystuje zaawansowane algorytmy całkowania równań różniczkowych
    zwyczajnych (ODE) do generowania trajektorii stanów systemu w odpowiedzi
    na zadany sygnał wymuszenia.
    """

    def __init__(self, obj: DynamicObject, u_func, dt=0.1):
        """
        Inicjalizuje generator danych dla konkretnego obiektu i sygnału sterującego.

        Args:
            obj (DynamicObject): Instancja klasy obiektu (np. CoupledTanks1),
                która definiuje równania stanu dx/dt.
            u_func (callable): Funkcja czasu zwracająca wartość sterowania u(t).
                Może to być funkcja stała, sinusoidalna lub losowa (APRBS).
            dt (float): Krok próbkowania (odstęp czasu między kolejnymi
                punktami w wygenerowanym zbiorze danych).
        """
        self.obj = obj
        self.u_func = u_func
        self.dt = dt
        # t_eval definiuje punkty, w których solver zapisze wyniki symulacji
        self.t_eval = np.arange(obj.t_span[0], obj.t_span[1], dt)

    def generate(self):
        """
        Przeprowadza proces całkowania numerycznego metodą Rungego-Kutty (RK45).

        Metoda ta "rozwiązuje" dynamikę obiektu, wyznaczając wartości stanów
        w kolejnych chwilach czasu, a następnie pakuje wyniki w strukturę SystemData.

        Returns:
            SystemData: Obiekt zawierający kompletny zbiór danych:
                - y: Macierz stanów wyjściowych (np. h1, h2) [N x wymiar_stanu]
                - u: Wektor/macierz zastosowanych sterowań [N x wymiar_u]
                - t: Wektor czasu symulacji [N]
        """

        def wrapped_f(t, x):
            """
            Funkcja pomocnicza (wrapper) dostosowująca interfejs obiektu
            do wymagań biblioteki scipy.integrate.

            Wstrzykuje wartość sygnału sterującego u(t) do równań różniczkowych
            obiektu w każdym kroku obliczeniowym solvera.
            """
            u = self.u_func(t)
            return self.obj.f(t, x, u)

        # Wywołanie profesjonalnego solvera o zmiennym kroku czasowym
        sol = solve_ivp(
            wrapped_f,
            self.obj.t_span,
            self.obj.x0,
            t_eval=self.t_eval,
            method='RK45'  # Metoda Rungego-Kutty 4. i 5. rzędu
        )

        # Wyznaczenie wartości sterowania u dla każdej chwili czasu t z rozwiązania
        # Jest to niezbędne, aby zestawić u(k) z y(k) w zbiorze treningowym
        u_values = np.array([self.u_func(ti) for ti in sol.t])

        # Pakowanie danych do ustandaryzowanego kontenera
        # sol.y.T transponuje macierz wyników do formatu [próbki x stany]
        return SystemData(sol.y.T, u_values, sol.t)