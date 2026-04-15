import matplotlib.pyplot as plt
import numpy as np


class SystemPlotter:
    @staticmethod
    def plot(t, u, y_true, dy_dt_true, y_sim_list=None, dy_dt_sim_list=None, legend_sim=None,
                        title="System Dynamics Comparison"):
        """
        Rozbudowana wizualizacja porównująca wiele symulacji na 5 subplotach.

        Args:
            t: wektor czasu
            u: sygnał wejściowy (sterowanie)
            y_true: macierz [N x 2] z prawdziwymi stanami (h1, h2)
            dy_dt_true: macierz [N x 2] z prawdziwymi pochodnymi
            y_sim_list: lista macierzy [N x 2] z wynikami różnych symulacji (opcjonalne)
            dy_dt_sim_list: lista macierzy [N x 2] z przewidzianymi pochodnymi (opcjonalne)
            legend_sim: lista nazw dla symulacji (np. ["MLP PyTorch", "MLP Manual"]) (opcjonalne)
            title: Tytuł główny
        """

        # Inicjalizacja list, jeśli nie zostały podane
        if y_sim_list is None: y_sim_list = []
        if dy_dt_sim_list is None: dy_dt_sim_list = []
        if legend_sim is None: legend_sim = []

        # Upewniamy się, że mamy listy, nawet jeśli przekazano pojedynczy wynik
        if not isinstance(y_sim_list, list): y_sim_list = [y_sim_list]
        if not isinstance(dy_dt_sim_list, list): dy_dt_sim_list = [dy_dt_sim_list]

        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Kolory dla różnych symulacji (żeby łatwo odróżnić algorytmy)
        true_colors = ['g', 'g', 'g', 'g']
        sim_colors = ['--r', '--b', '--m', '--c']

        # 1. Sygnał wejściowy u(t)
        axes[0].step(t, u[:, 0] if u.ndim > 1 else u, 'k', where='post', label="u(t) [V]")
        axes[0].set_title("Sygnał wejściowy")
        axes[0].set_ylabel("u")

        # 2. Stan h1
        axes[1].plot(t, y_true[:, 0], true_colors[0], linewidth=2, alpha=0.9, label="h1 (true)")
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[1].plot(t, y_sim[:, 0], sim_colors[i % len(sim_colors)], alpha=0.7, label=f"h1 ({label})")
        axes[1].set_title("Poziom h1")
        axes[1].set_ylabel("h1 [cm]")

        # 3. Stan h2
        axes[2].plot(t, y_true[:, 1], true_colors[1], linewidth=2, alpha=0.9, label="h2 (true)")
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[2].plot(t, y_sim[:, 1], sim_colors[i % len(sim_colors)], alpha=0.7, label=f"h2 ({label})")
        axes[2].set_title("Poziom h2")
        axes[2].set_ylabel("h2 [cm]")

        # 4. Pochodna dh1/dt
        axes[3].plot(t, dy_dt_true[:, 0], true_colors[2], alpha=0.9, label="dh1/dt (true)")
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[3].plot(t, dy_dt[:, 0], sim_colors[i % len(sim_colors)], alpha=0.7, label=f"dh1/dt ({label})")
        axes[3].set_title("Pochodna dh1/dt")
        axes[3].set_ylabel("dh1/dt")

        # 5. Pochodna dh2/dt
        axes[4].plot(t, dy_dt_true[:, 1], true_colors[3], alpha=0.9, label="dh2/dt (true)")
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[4].plot(t, dy_dt[:, 1], sim_colors[i % len(sim_colors)], alpha=0.7, label=f"dh2/dt ({label})")
        axes[4].set_title("Pochodna dh2/dt")
        axes[4].set_ylabel("dh2/dt")
        axes[4].set_xlabel("Czas [s]")

        # Dodatki estetyczne
        for ax in axes:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show(block=False)

    def plot_noise_comparison(t, u, y_true, dy_dt_true, y_noise=None, dy_dt_noise=None,
                              noise_label="Data with Noise", title="Data Integrity Check: Clean vs Noisy"):
        """
        Wizualizacja porównująca dane czyste (True) z zaszumionymi (Noise) na 5 subplotach.
        Wszystkie linie są ciągłe.
        """

        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Kolory: b/g dla prawdy, orange/red dla szumu (bardziej widoczne niż przerywane)
        color_h1_true, color_h1_noise = 'blue', 'blue'
        color_h2_true, color_h2_noise = 'green', 'green'

        # 1. Sygnał wejściowy u(t)
        axes[0].step(t, u[:, 0] if u.ndim > 1 else u, 'black', where='post', label="u(t) [V]")
        axes[0].set_title("Sygnał wejściowy")
        axes[0].set_ylabel("u")

        # 2. Stan h1
        axes[1].plot(t, y_true[:, 0], color=color_h1_true, linewidth=2, label="h1 (Clean)")
        if y_noise is not None:
            axes[1].plot(t, y_noise[:, 0], color=color_h1_noise, linewidth=1, alpha=0.5, label=f"h1 ({noise_label})")
        axes[1].set_title("Poziom h1")
        axes[1].set_ylabel("h1 [cm]")

        # 3. Stan h2
        axes[2].plot(t, y_true[:, 1], color=color_h2_true, linewidth=2, label="h2 (Clean)")
        if y_noise is not None:
            axes[2].plot(t, y_noise[:, 1], color=color_h2_noise, linewidth=1, alpha=0.5, label=f"h2 ({noise_label})")
        axes[2].set_title("Poziom h2")
        axes[2].set_ylabel("h2 [cm]")

        # 4. Pochodna dh1/dt
        axes[3].plot(t, dy_dt_true[:, 0], color=color_h1_true, label="dh1/dt (Clean)")
        if dy_dt_noise is not None:
            axes[3].plot(t, dy_dt_noise[:, 0], color=color_h1_noise, linewidth=1, alpha=0.5, label=f"dh1/dt ({noise_label})")
        axes[3].set_title("Pochodna dh1/dt")
        axes[3].set_ylabel("dh1/dt")

        # 5. Pochodna dh2/dt
        axes[4].plot(t, dy_dt_true[:, 1], color=color_h2_true, label="dh2/dt (Clean)")
        if dy_dt_noise is not None:
            axes[4].plot(t, dy_dt_noise[:, 1], color=color_h2_noise, linewidth=1, alpha=0.5, label=f"dh2/dt ({noise_label})")
        axes[4].set_title("Pochodna dh2/dt")
        axes[4].set_ylabel("dh2/dt")
        axes[4].set_xlabel("Czas [s]")

        # Dodatki estetyczne
        for ax in axes:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show(block=False)

# --- PRZYKŁAD UŻYCIA ---
# plotter = SystemPlotter()
# plotter.plot_comparison(
#     t=t_test,
#     u=u_test,
#     y_true=y_true,
#     dy_dt_true=dy_dt_true,
#     y_sim_list=[y_sim_torch, y_sim_manual],
#     dy_dt_sim_list=[dy_dt_pred_torch, dy_dt_pred_manual],
#     legend_sim=["MLP PyTorch", "MLP Własne"],
#     title="Porównanie dwóch implementacji sieci"
# )