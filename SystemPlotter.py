import matplotlib.pyplot as plt
import numpy as np


class SystemPlotter:
    """
    Klasa do wizualizacji trajektorii systemu dynamicznego.
    Obsługuje dane rzeczywiste i symulowane.
    """

    @staticmethod
    def plot(t, y_true=None, dy_dt_true=None,
             y_sim=None, dy_dt_sim=None, u=None, title="System dynamics"):
        """
        Tworzy wykres porównawczy.

        Args:
            t (np.ndarray): wektor czasu
            y_true (np.ndarray, opcjonalnie): rzeczywista trajektoria y(t)
            dy_dt_true (np.ndarray, opcjonalnie): rzeczywiste dy/dt
            y_sim (np.ndarray, opcjonalnie): symulowana trajektoria y(t)
            dy_dt_sim (np.ndarray, opcjonalnie): dy/dt przewidziane przez MLP
            u (np.ndarray, opcjonalnie): wektor wymuszenia u(t)
            title (str): tytuł wykresu
        """
        plt.figure(figsize=(10,6))
        plt.title(title)

        if dy_dt_true is not None:
            plt.plot(t, dy_dt_true, label="dy/dt (true)")
        if dy_dt_sim is not None:
            plt.plot(t, dy_dt_sim, label="dy/dt (MLP)")
        if y_true is not None:
            plt.plot(t, y_true, ':', label='y(t) (true)')
        if y_sim is not None:
            plt.plot(t, y_sim, ':', label='y(t) (sim)')
        if u is not None:
            if u.ndim > 1 and u.shape[1] == 1:
                plt.plot(t, u[:,0], '--', label='u(t)')
            else:
                for i in range(u.shape[1]):
                    plt.plot(t, u[:,i], '--', label=f'u{i+1}(t)')

        plt.xlabel("t")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()