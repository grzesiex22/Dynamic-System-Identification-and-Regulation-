import matplotlib.pyplot as plt
import numpy as np


class SystemPlotter:
    @staticmethod
    def plot(t, u, y_true, dy_dt_true, y_sim_list, dy_dt_sim_list, legend_sim,
                        title="System Dynamics Comparison"):
        """
        Rozbudowana wizualizacja porównująca wiele symulacji na 5 subplotach.

        Args:
            t: wektor czasu
            u: sygnał wejściowy (sterowanie)
            y_true: macierz [N x 2] z prawdziwymi stanami (h1, h2)
            dy_dt_true: macierz [N x 2] z prawdziwymi pochodnymi
            y_sim_list: lista macierzy [N x 2] z wynikami różnych symulacji
            dy_dt_sim_list: lista macierzy [N x 2] z przewidzianymi pochodnymi
            legend_sim: lista nazw dla symulacji (np. ["MLP PyTorch", "MLP Manual"])
            title: Tytuł główny
        """
        # Upewniamy się, że mamy listy, nawet jeśli przekazano pojedynczy wynik
        if not isinstance(y_sim_list, list): y_sim_list = [y_sim_list]
        if not isinstance(dy_dt_sim_list, list): dy_dt_sim_list = [dy_dt_sim_list]

        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Kolory dla różnych symulacji (żeby łatwo odróżnić algorytmy)
        sim_colors = ['--r', '--g', '--m', '--c']

        # 1. Sygnał wejściowy u(t)
        axes[0].step(t, u[:, 0] if u.ndim > 1 else u, 'k', where='post', label="u(t) [V]")
        axes[0].set_title("Sygnał wejściowy")
        axes[0].set_ylabel("u")

        # 2. Stan h1
        axes[1].plot(t, y_true[:, 0], 'b', linewidth=2, label="h1 (true)")
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[1].plot(t, y_sim[:, 0], sim_colors[i % len(sim_colors)], label=f"h1 ({label})")
        axes[1].set_title("Poziom h1")
        axes[1].set_ylabel("h1 [cm]")

        # 3. Stan h2
        axes[2].plot(t, y_true[:, 1], 'g', linewidth=2, label="h2 (true)")
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[2].plot(t, y_sim[:, 1], sim_colors[i % len(sim_colors)], label=f"h2 ({label})")
        axes[2].set_title("Poziom h2")
        axes[2].set_ylabel("h2 [cm]")

        # 4. Pochodna dh1/dt
        axes[3].plot(t, dy_dt_true[:, 0], 'b', alpha=0.4, label="dh1/dt (true)")
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[3].plot(t, dy_dt[:, 0], sim_colors[i % len(sim_colors)], label=f"dh1/dt ({label})")
        axes[3].set_title("Pochodna dh1/dt")
        axes[3].set_ylabel("dh1/dt")

        # 5. Pochodna dh2/dt
        axes[4].plot(t, dy_dt_true[:, 1], 'g', alpha=0.4, label="dh2/dt (true)")
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[4].plot(t, dy_dt[:, 1], sim_colors[i % len(sim_colors)], label=f"dh2/dt ({label})")
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