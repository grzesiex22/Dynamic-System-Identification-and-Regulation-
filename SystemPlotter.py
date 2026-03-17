import matplotlib.pyplot as plt
import numpy as np


class SystemPlotter:
    """
    Narzędzie wizualizacyjne do analizy systemów dynamicznych wielowymiarowych.

    Klasa generuje wielopoziomowe wykresy porównawcze, zestawiając dane
    referencyjne (ground truth) z wynikami predykcji modelu neuronowego.
    Ułatwia ocenę jakości identyfikacji oraz stabilności symulacji.
    """

    @staticmethod
    def plot(t, y_true=None, dy_dt_true=None,
             y_sim=None, dy_dt_sim=None, u=None, title="System Dynamics Identification"):
        """
        Generuje zestaw wykresów czasowych dla stanów, ich pochodnych oraz wymuszeń.

        Metoda automatycznie dostosowuje liczbę podwykresów (subplots) do
        przekazanych danych. Wykorzystuje wspólną oś czasu (sharex) dla lepszej
        czytelności korelacji między sygnałami.

        Args:
            t (np.ndarray): Wektor czasu [N].
            y_true (np.ndarray, opcjonalnie): Rzeczywiste trajektorie stanów [N x m].
            dy_dt_true (np.ndarray, opcjonalnie): Rzeczywiste pochodne stanów [N x m].
            y_sim (np.ndarray, opcjonalnie): Trajektorie przewidziane przez MLP [N x m].
            dy_dt_sim (np.ndarray, opcjonalnie): Pochodne przewidziane przez MLP [N x m].
            u (np.ndarray, opcjonalnie): Sygnał sterujący [N x k].
            title (str): Główny tytuł okna wykresu.
        """

        # 1. Dynamiczne obliczanie liczby wymaganych wierszy wykresu
        rows = 0
        if y_true is not None or y_sim is not None: rows += 1
        if dy_dt_true is not None or dy_dt_sim is not None: rows += 1
        if u is not None: rows += 1

        # Tworzenie figury i osi
        fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows), sharex=True)
        # Zapewnienie, że axes jest zawsze iterowalne (nawet przy 1 rzędzie)
        if rows == 1: axes = [axes]

        curr_ax = 0
        fig.suptitle(title, fontsize=16)

        # --- SEKCOJA 1: POZIOMY STANÓW (y / h) ---
        # Porównanie rzeczywistych i symulowanych wartości h1 i h2
        if y_true is not None or y_sim is not None:
            ax = axes[curr_ax]
            if y_true is not None:
                ax.plot(t, y_true[:, 0], 'b-', alpha=0.6, label='h1 (true)')
                ax.plot(t, y_true[:, 1], 'g-', alpha=0.6, label='h2 (true)')
            if y_sim is not None:
                ax.plot(t, y_sim[:, 0], 'b--', linewidth=2, label='h1 (MLP)')
                ax.plot(t, y_sim[:, 1], 'g--', linewidth=2, label='h2 (MLP)')
            ax.set_ylabel("Level [cm]")
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.7)
            curr_ax += 1

        # --- SEKCJA 2: DYNAMIKA ZMIAN (dy/dt) ---
        # Wizualizacja prędkości zmian stanów (pochodne)
        if dy_dt_true is not None or dy_dt_sim is not None:
            ax = axes[curr_ax]
            if dy_dt_true is not None:
                ax.plot(t, dy_dt_true[:, 0], 'c-', alpha=0.5, label='dh1/dt (true)')
                ax.plot(t, dy_dt_true[:, 1], 'm-', alpha=0.5, label='dh2/dt (true)')
            if dy_dt_sim is not None:
                ax.plot(t, dy_dt_sim[:, 0], 'c--', label='dh1/dt (MLP)')
                ax.plot(t, dy_dt_sim[:, 1], 'm--', label='dh2/dt (MLP)')
            ax.set_ylabel("Rate [cm/s]")
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.7)
            curr_ax += 1

        # --- SEKCJA 3: SYGNAŁY STERUJĄCE (u) ---
        # Wykres wymuszenia przedstawiony jako funkcja schodkowa
        if u is not None:
            ax = axes[curr_ax]
            # Obsługa u jako wektora 1D lub macierzy kolumnowej
            u_plot = u if u.ndim == 1 else u[:, 0]
            ax.step(t, u_plot, 'r', where='post', label='u(t) [V]')
            ax.set_ylabel("Input [V]")
            ax.set_xlabel("Time [s]")
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.7)

        # Optymalizacja układu wykresów
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()