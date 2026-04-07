import numpy as np
from Objects.DynamicObject import DynamicObject


class CoupledTanks1(DynamicObject):
    """
    Model matematyczny stacjonarnego układu dwóch zbiorników kaskadowych.

    System modeluje dwa nieliniowe zbiorniki, w których parametry fizyczne są
    niezmienne w czasie. Służy jako obiekt bazowy do treningu i wstępnej
    identyfikacji dynamiki.
    """

    def __init__(self, t_end=500):
        """
        Inicjalizuje parametry fizyczne i warunki brzegowe systemu stacjonarnego.

        Attributes:
            A1, A2 (float): Powierzchnie przekroju zbiorników [cm^2].
            a1, a2 (float): Powierzchnie otworów wypływowych [cm^2].
            g (float): Przyspieszenie ziemskie [cm/s^2].
            k (float): Stała wydajności pompy.
        """
        super().__init__()
        self.x0 = [0.0, 0.0]  # Poziomy początkowe [h1, h2]
        self.t_span = (0, t_end)  # Domyślny czas symulacji [s]

        # Parametry konstrukcyjne
        self.A1, self.A2 = 100.0, 100.0
        self.a1, self.a2 = 0.5, 0.4
        self.g = 981.0
        self.k = 2.5

    def f(self, t, x, u):
        """
        Oblicza pochodne stanów dla obiektu stacjonarnego.

        Args:
            t (float): Czas symulacji.
            x (list): Wektor stanu [h1, h2].
            u (float): Sygnał sterujący (napięcie pompy).

        Returns:
            list: Wektor pochodnych [dh1/dt, dh2/dt].
        """
        h1, h2 = x

        # Zabezpieczenie numeryczne przed ujemnymi poziomami wody
        h1 = max(h1, 0)
        h2 = max(h2, 0)

        # Równania bilansu masy (nieliniowość pierwiastkowa)
        dh1dt = (1 / self.A1) * (self.k * u - self.a1 * np.sqrt(2 * self.g * h1))
        dh2dt = (1 / self.A2) * (self.a1 * np.sqrt(2 * self.g * h1) - self.a2 * np.sqrt(2 * self.g * h2))

        return [dh1dt, dh2dt]


class CoupledTanks2(DynamicObject):
    """
    Model matematyczny niestacjonarnego układu dwóch zbiorników kaskadowych.

    Obiekt symuluje zmianę właściwości fizycznych systemu w trakcie pracy (np. awarię
    lub zatkanie odpływu). Wykorzystywany do testowania adaptacyjnych zdolności
    modeli uczenia maszynowego.
    """

    def __init__(self):
        """
        Inicjalizuje parametry fizyczne systemu z możliwością ich zmiany w czasie.
        Parametry konstrukcyjne są identyczne jak w wersji stacjonarnej CoupledTanks1.
        """
        super().__init__()
        self.x0 = [0.0, 0.0]
        self.t_span = (0, 500)

        self.A1, self.A2 = 100.0, 100.0
        self.a1, self.a2 = 0.5, 0.4
        self.g = 981.0
        self.k = 2.5

    def f(self, t, x, u):
        """
        Oblicza pochodne stanów z uwzględnieniem zmiany parametrów (niestacjonarność).

        W 250. sekundzie symulacji następuje skokowa zmiana przekroju otworu
        wypływowego drugiego zbiornika (a2), co modeluje np. częściowe zatkanie rury.

        Args:
            t (float): Czas symulacji.
            x (list): Wektor stanu [h1, h2].
            u (float): Sygnał sterujący.

        Returns:
            list: Wektor pochodnych [dh1/dt, dh2/dt].
        """
        h1, h2 = x

        # --- MECHANIZM NIESTACJONARNOŚCI ---
        # Symulacja zmiany parametru obiektu w czasie rzeczywistym
        current_a2 = self.a2
        if t > 250:
            current_a2 = self.a2 * 0.5  # Zmniejszenie prześwitu o 50%
        # -----------------------------------

        h1 = max(h1, 0)
        h2 = max(h2, 0)

        dh1dt = (1 / self.A1) * (self.k * u - self.a1 * np.sqrt(2 * self.g * h1))
        dh2dt = (1 / self.A2) * (self.a1 * np.sqrt(2 * self.g * h1) - current_a2 * np.sqrt(2 * self.g * h2))

        return [dh1dt, dh2dt]