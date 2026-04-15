import numpy as np
from scipy.integrate import solve_ivp
from Generators.SystemData import SystemData
from Objects.DynamicObject import DynamicObject


class DataGenerator:
    """
    Klasa odpowiedzialna za symulację numeryczną obiektów dynamicznych (Silnik Symulacyjny).

    Wykorzystuje zaawansowane algorytmy całkowania równań różniczkowych zwyczajnych (ODE)
    do generowania trajektorii stanów systemu (tzw. Ground Truth) w odpowiedzi na zadany
    sygnał wymuszenia u(t).
    """

    def __init__(self, obj: DynamicObject, u_func, dt=0.1):
        """
        Inicjalizuje generator danych dla konkretnego obiektu i sygnału sterującego.

        Args:
            obj (DynamicObject): Instancja klasy obiektu (np. CoupledTanks1),
                definiująca fizykę systemu poprzez funkcję przejścia f(t, x, u).
            u_func (callable): Funkcja czasu (np. obiekt klasy APRBSSignal),
                zwracająca wartość sterowania u dla zadanej chwili t.
            dt (float): Krok próbkowania wyjściowego zbioru danych [s].
            noise_level (float): Odchylenie standardowe (sigma) szumu Gaussa.
                                             0.0 oznacza brak szumu.
        """

        self.obj = obj
        self.u_func = u_func
        self.dt = dt
        # t_eval definiuje punkty, w których solver zapisze wyniki symulacji
        self.t_eval = np.arange(obj.t_span[0], obj.t_span[1], dt)

    def generate(self):
        """
        Przeprowadza proces całkowania numerycznego metodą adaptacyjną Rungego-Kutty (RK45).

        Metoda rozwiązuje problem początkowy (Initial Value Problem), wyznaczając
        ewolucję stanów obiektu. Wynik jest synchronizowany z wymuszeniem u i pakowany
        w ustandaryzowany kontener SystemData.

        Returns:
            SystemData: Obiekt zawierający zsynchronizowane trajektorie:
                - y (np.ndarray): Macierz stanów wyjściowych (np. h1, h2) [N x dim_y].
                - u (np.ndarray): Macierz zastosowanych sterowań [N x dim_u].
                - t (np.ndarray): Wektor czasu symulacji [N].
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

        # Wywołanie solvera z adaptacyjnym krokiem czasowym (RK45)
        sol = solve_ivp(
            wrapped_f,
            self.obj.t_span,
            self.obj.x0,
            t_eval=self.t_eval,
            method='RK45',  # Adaptacyjna metoda Rungego-Kutty 4/5 rzędu
            rtol=1e-6,  # Tolerancja względna (wysoka precyzja)
            atol=1e-9  # Tolerancja bezwzględna
        )

        # Pobieramy czyste stany (Ground Truth)
        y_values = sol.y.T  # Macierz [N x 2] (h1, h2)

        # Synchronizacja: wyznaczenie wartości sterowania dokładnie dla tych chwil,
        # które zwrócił solver. To kluczowe dla poprawnego treningu sieci MLP.
        u_values = np.array([self.u_func(ti) for ti in sol.t])

        # Pakowanie wyników do kontenera SystemData.
        # sol.y jest transponowane z [dim_y x N] na [N x dim_y].
        return SystemData(y_values, u_values, sol.t)

    def add_noise(self, system_data: SystemData, noise_level=0.1):
        """
        Przeprowadza proces całkowania numerycznego metodą adaptacyjną Rungego-Kutty (RK45).

        Metoda rozwiązuje problem początkowy (Initial Value Problem), wyznaczając
        ewolucję stanów obiektu. Wynik jest synchronizowany z wymuszeniem u i pakowany
        w ustandaryzowany kontener SystemData.

        Returns:
            SystemData: Obiekt zawierający zsynchronizowane trajektorie:
                - y (np.ndarray): Macierz stanów wyjściowych (np. h1, h2) [N x dim_y].
                - u (np.ndarray): Macierz zastosowanych sterowań [N x dim_u].
                - t (np.ndarray): Wektor czasu symulacji [N].
        """

        # Pobieramy czyste stany (Ground Truth)
        y_values = system_data.y  # Macierz [N x 2] (h1, h2)

        # --- Dodawanie szumu pomiarowego ---

        # Generujemy szum o rozkładzie normalnym (Gauss) dla każdego punktu pomiarowego
        # np.random.normal(lokalizacja, skala, rozmiar)
        noise = np.random.normal(0, noise_level, y_values.shape)
        y_values = y_values + noise

        # Zapewnienie, że wysokość nie spadnie poniżej 0 przez szum
        y_values = np.maximum(y_values, 0)

        # Pakowanie wyników do kontenera SystemData.
        # sol.y jest transponowane z [dim_y x N] na [N x dim_y].
        return SystemData(y_values, system_data.u, system_data.t)
