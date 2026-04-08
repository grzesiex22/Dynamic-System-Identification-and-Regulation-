import numpy as np
from matplotlib import pyplot as plt


class BaseSignal:
    """
    Klasa bazowa dla generatorów sygnałów pobudzających.
    Definiuje podstawowe parametry czasowe i zakresy amplitud.
    """
    def __init__(self, t_end, dt, amp_range=(2, 8)):
        """
        Inicjalizacja parametrów sygnału.

        Args:
            t_end (float): Czas końcowy sygnału [s].
            dt (float): Krok próbkowania [s].
            amp_range (tuple): Zakres dopuszczalnych napięć (min, max) [V].
        """
        self.t_end = t_end
        self.dt = dt
        self.amin, self.amax = amp_range
        self.t = np.arange(0, t_end, dt)


class APRBSSignal(BaseSignal):
    """
    Sygnał APRBS (Amplitude Pseudo-Random Binary Sequence).
    Charakteryzuje się skokowymi zmianami amplitudy utrzymywanymi przez losowy czas.
    Idealny do uczenia sieci stanów ustalonych oraz odpowiedzi skokowej.
    """

    def __init__(self, t_end, dt, amp_range=(2, 8), hold_range=(50, 200)):
        """
        Args:
            hold_range (tuple): Zakres (min, max) czasu utrzymania jednej wartości (w krokach dt).
        """

        super().__init__(t_end, dt, amp_range)
        self.hold_range = hold_range
        # Generujemy tablicę wartości raz w konstruktorze
        self.u_values = self._create_aprbs_array()

    def _create_aprbs_array(self):
        """Generuje tablicę sygnału metodą Zero-Order Hold."""

        u = np.zeros_like(self.t)
        i = 0
        while i < len(self.t):
            # Losujemy jak długo trzymać sygnał
            hold = np.random.randint(*self.hold_range)
            # Losujemy amplitudę dla tego segmentu
            amplitude = np.random.uniform(self.amin, self.amax)

            # Wypełniamy fragment tablicy
            u[i: i + hold] = amplitude
            i += hold
        return u[:len(self.t)]

    def generate(self):
        """Zwraca wygenerowaną tablicę (do debugowania/wykresów)."""
        return self.u_values

    def __call__(self, t):
        """
        Metoda dla solvera. Realizuje tzw. Zero-Order Hold (ZOH).
        Zwraca wartość z tablicy dla najbliższego kroku czasowego 't'.
        """
        # Obliczamy indeks na podstawie czasu t i kroku dt
        idx = int(t / self.dt)

        # Zabezpieczenie przed przekroczeniem zakresu tablicy (np. na samym końcu t_end)
        idx = max(0, min(idx, len(self.u_values) - 1))

        return self.u_values[idx]


class MultisineSignal(BaseSignal):
    """
    Sygnał wieloczęstotliwościowy (Multisine).
    Suma wielu sinusojid o losowych częstotliwościach i fazach.
    Uczy sieć dynamiki w dziedzinie częstotliwości (np. jak szybko system reaguje).
    """

    def __init__(self, t_end, dt, amp_range=(2, 8), n_freqs=10, f_max=0.02):
        """
        Args:
            n_freqs (int): Liczba sumowanych składowych harmonicznych.
            f_max (float): Maksymalna częstotliwość składowej [Hz].
        """

        super().__init__(t_end, dt, amp_range)
        self.freqs = np.random.uniform(0.00001, f_max, n_freqs)
        self.phases = np.random.uniform(0, 2 * np.pi, n_freqs)

    def generate(self):
        """Generuje pełną trajektorię sygnału dla wektora czasu self.t."""
        return np.array([self(ti) for ti in self.t])

    def __call__(self, t):
        """Wyznacza sumę sinusów znormalizowaną do zakresu amp_range."""

        # 1. Liczymy sumę sinusów dla konkretnego punktu czasu 't'
        u_raw = 0.0
        for f, p in zip(self.freqs, self.phases):
            u_raw += np.sin(2 * np.pi * f * t + p)

        # 2. Normalizacja "teoretyczna"
        # Suma N sinusów o amplitudzie 1 zawsze mieści się w zakresie [-N, N]
        n = len(self.freqs)
        u_norm = (u_raw + n) / (2 * n)  # Przesunięcie do zakresu [0, 1]

        # 3. Skalowanie do amp_range
        return u_norm * (self.amax - self.amin) + self.amin


class FilteredNoiseSignal(BaseSignal):
    """
    Przefiltrowany Szum Biały (Low-pass Filtered Noise).
    Sygnał chaotyczny o ograniczonym paśmie. Pomaga sieci uzyskać odporność
    na szumy procesowe i modelować nieprzewidywalne zmiany sterowania.
    """

    def __init__(self, t_end, dt, amp_range=(2, 8), cutoff=0.001):
        """
        Args:
            cutoff (float): Częstotliwość odcięcia filtru dolnoprzepustowego (współczynnik bezwładności).
        """

        super().__init__(t_end, dt, amp_range)
        self.cutoff = cutoff
        # Generujemy bazowy, przefiltrowany wektor szumu
        self.u_values = self._create_filtered_noise()

    def _create_filtered_noise(self):
        """Implementuje filtr dolnoprzepustowy EMA (Exponential Moving Average)."""

        noise = np.random.randn(len(self.t))
        u = np.zeros_like(noise)

        # 1. Obliczamy współczynnik alpha (filtr dolnoprzepustowy)
        alpha = 2 * np.pi * self.cutoff * self.dt / (2 * np.pi * self.cutoff * self.dt + 1)

        # 2. Iteracyjne filtrowanie (Exponential Moving Average)
        for i in range(1, len(noise)):
            u[i] = u[i - 1] + alpha * (noise[i] - u[i - 1])

        # 3. Normalizacja "w locie" do zakresu [self.amin, self.amax]
        u_min = u.min()
        u_max = u.max()

        # Zabezpieczenie: jeśli sygnał jest stały, zwróć dolną granicę
        if abs(u_max - u_min) < 1e-9:
            return np.full_like(u, self.amin)

        # Skalowanie do [0, 1]
        u_norm = (u - u_min) / (u_max - u_min)

        # Skalowanie do docelowego zakresu fizycznego [V]
        return u_norm * (self.amax - self.amin) + self.amin

    def generate(self):
        """Zwraca wygenerowaną tablicę przefiltrowanego szumu."""
        return self.u_values

    def __call__(self, t):
        """Zwraca interpolowaną wartość sygnału dla gładkiego przejścia między krokami."""
        return np.interp(t, self.t, self.u_values)


# # Generowanie próbek
# gen1 = APRBSSignal(t_end=1000, dt=0.5, amp_range=(3, 9))
# u_aprbs = gen1.generate()
# gen2 = MultisineSignal(t_end=1000, dt=0.5, amp_range=(3, 9))
# u_sine = gen2.generate()
# gen3 = FilteredNoiseSignal(t_end=1000, dt=0.5, amp_range=(3, 9))
# u_noise = gen3.generate()
#
#
# # Wykresy
# plt.figure(figsize=(12, 8))
#
# plt.subplot(3, 1, 1)
# plt.step(gen1.t, u_aprbs, 'r', where='post')
# plt.title("APRBS (Skoki napięcia - uczy stanów ustalonych)")
# plt.grid(True, alpha=0.3)
#
# plt.subplot(3, 1, 2)
# plt.plot(gen2.t, u_sine, 'b')
# plt.title("Multisine (Gładkie fale - uczy dynamiki częstotliwościowej)")
# plt.grid(True, alpha=0.3)
#
# plt.subplot(3, 1, 3)
# plt.plot(gen3.t, u_noise, 'g')
# plt.title("Filtered Noise (Szum - uczy odporności na chaos)")
# plt.xlabel("Czas [s]")
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()