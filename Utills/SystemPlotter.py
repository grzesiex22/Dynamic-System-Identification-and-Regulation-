import os

import matplotlib.pyplot as plt
import numpy as np


class SystemPlotter:
    @staticmethod
    def plot(t, u, y_true, dy_dt_true, y_sim_list=None, dy_dt_sim_list=None, legend_sim=None,
             title="System Dynamics Comparison", save_name=None, dataset=None, folder="Results",
             show=True, x_lim=None):
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
            save_name (str): nazwa pliku
            dataset (str): dataset
            folder (str): folder
            show (bool): jeśli True to wyświetli wykres, jeśli False to nie
        """
        # Jeśli użytkownik podał x_lim (np. [0, 500]), przycinamy dane na wejściu
        if x_lim is not None:
            start, end = x_lim
            # Zakładamy, że x_lim podano w sekundach, więc przeliczamy na indeksy
            t_mask = (t >= start) & (t <= end)

            t = t[t_mask]
            u = u[t_mask]
            y_true = y_true[t_mask]
            dy_dt_true = dy_dt_true[t_mask]

            # Przycinamy listy symulacji (każdy model w pętli)
            y_sim_list = [sim[t_mask] for sim in y_sim_list]
            dy_dt_sim_list = [sim[t_mask] for sim in dy_dt_sim_list]

        # Inicjalizacja list, jeśli nie zostały podane
        if y_sim_list is None: y_sim_list = []
        if dy_dt_sim_list is None: dy_dt_sim_list = []
        if legend_sim is None: legend_sim = []

        # Upewniamy się, że mamy listy, nawet jeśli przekazano pojedynczy wynik
        if not isinstance(y_sim_list, list): y_sim_list = [y_sim_list]
        if not isinstance(dy_dt_sim_list, list): dy_dt_sim_list = [dy_dt_sim_list]

        fig, axes = plt.subplots(
            5, 1,
            figsize=(12, 18),
            sharex=True,
            gridspec_kw={'height_ratios': [0.4, 1.15, 1.15, 1.15, 1.15]}  # u(t) jest niższy, reszta większa
        )
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Kolory (użycie palety profesjonalnej)
        sim_colors = [
            '#FF0000',  # Intensywna Czerwień (np. dla Torch_MLP)
            '#0055FF',  # Głęboki Niebieski (np. dla Own_MLP)
            '#00AA00',  # Wyrazista Zieleń (np. dla Torch_LSTM)
            '#FF8C00',  # Ciemny Pomarańcz (np. dla Ridge)
            '#9400D3',  # Ciemny Fiolet
            '#00CED1'  # Mocny Turkus
        ]

        # Bardzo ciemny grafit dla prawdy
        true_color = '#1A1A1A'

        # 1. Sygnał wejściowy u(t)
        axes[0].step(t, u[:, 0] if u.ndim > 1 else u, 'k', where='post', label="u(t) [V]")
        axes[0].set_title("Sygnał wejściowy")
        axes[0].set_ylabel("u")

        # 2. Stan h1
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[1].plot(t, y_sim[:, 0], color=sim_colors[i % len(sim_colors)], linewidth=1.0, alpha=0.8,
                         label=label, zorder=2)
        axes[1].plot(t, y_true[:, 0], color=true_color, linewidth=2.5, alpha=0.5, label="True", zorder=10)
        axes[1].set_title("Poziom h1")
        axes[1].set_ylabel("h1 [cm]")

        # 3. Stan h2
        for i, y_sim in enumerate(y_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[2].plot(t, y_sim[:, 1], color=sim_colors[i % len(sim_colors)], linewidth=1.0, alpha=0.8,
                         label=label, zorder=2)
        axes[2].plot(t, y_true[:, 1], color=true_color, linewidth=2.5, alpha=0.5, label="True", zorder=10)
        axes[2].set_title("Poziom h2")
        axes[2].set_ylabel("h2 [cm]")

        # 4. Pochodna dh1/dt
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[3].plot(t, dy_dt[:, 0], color=sim_colors[i % len(sim_colors)], linewidth=1.0, alpha=0.8,
                         label=label, zorder=2)
        axes[3].plot(t, dy_dt_true[:, 0], color=true_color, linewidth=2.5, alpha=0.5, label="True", zorder=10)
        axes[3].set_title("Pochodna dh1/dt")
        axes[3].set_ylabel("dh1/dt")

        # 5. Pochodna dh2/dt
        for i, dy_dt in enumerate(dy_dt_sim_list):
            label = legend_sim[i] if i < len(legend_sim) else f"Sim {i + 1}"
            axes[4].plot(t, dy_dt[:, 1], color=sim_colors[i % len(sim_colors)], linewidth=1.0, alpha=0.8,
                         label=label, zorder=2)
        axes[4].plot(t, dy_dt_true[:, 1], color=true_color, linewidth=2.5, alpha=0.5, label="True", zorder=10)
        axes[4].set_title("Pochodna dh2/dt")
        axes[4].set_ylabel("dh2/dt")
        axes[4].set_xlabel("Czas [s]")

        # Dodatki estetyczne
        for ax in axes:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # --- MODUŁ ZAPISU ---
        if save_name and dataset and folder:
            save_path = SystemPlotter.save(plt_=plt, save_name=save_name, dataset=dataset, folder=folder)
            print(f"    💾 Wykres zapisany: {save_path}")

        plt.show(block=False)

        if not show:
            plt.close('all')

    def plot_noise_comparison(t, u, y_true, dy_dt_true, y_noise=None, dy_dt_noise=None,
                              noise_label="Data with Noise", title="Data Integrity Check: Clean vs Noisy",
                              save_name=None, dataset=None, folder="Results", show=True):
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

        # --- MODUŁ ZAPISU ---
        if save_name and dataset and folder:
            save_path = SystemPlotter.save(plt_=plt, save_name=save_name, dataset=dataset, folder=folder)
            print(f"    💾 Wykres zapisany: {save_path}")

        plt.show(block=False)

        if not show:
            plt.close('all')

    @staticmethod
    def plot_learning_curves(history, model_name="Model", dataset="", v_name="CLEAN", save_name="",
                             folder="Results", show=True):
        if not history or "train_loss" not in history:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history["train_loss"]) + 1)

        # --- Tłumaczenie wariantu na polski ---
        if v_name.upper() == "CLEAN":
            variant_desc = "Dane czyste"
        else:
            # Zamienia NOISY_0.01 na Szum: 0.01
            variant_desc = v_name.replace("NOISY_", "Szum: ").replace("_", ".")

        # Rysowanie głównych krzywych
        plt.plot(epochs, history["train_loss"], 'b', label='Trening')
        plt.plot(epochs, history["val_loss"], 'r', label='Walidacja')

        # Obsługa "Best Epoch"
        best_ep = history.get("best_epoch", 0)

        # Sprawdzamy, czy best_ep jest sensowny (indeksowanie od 1)
        if 0 < best_ep <= len(history["val_loss"]):
            best_val_loss = history["val_loss"][best_ep - 1]

            # Pionowa linia przerywana do osi X
            plt.axvline(x=best_ep, color='green', linestyle='--', alpha=0.5, linewidth=1)

            # Zielona kropka w punkcie najlepszego wyniku
            plt.plot(best_ep, best_val_loss, 'go', markersize=8, label=f'Najlepsza epoka ({best_ep})')

            # Napis przy kropce (Epoch i Value)
            # Przesunięcie (offset) tekstu, żeby nie nachodził na kropkę
            plt.text(best_ep, best_val_loss,
                     f'  Epoka: {best_ep}\n  Strata: {best_val_loss:.8f}',
                     color='green', fontweight='bold', va='bottom', ha='left')

        # Estetyka wykresu
        plt.yscale('log')  # Logarytmiczna oś Y dla lepszej widoczności
        plt.title(f"Krzywa uczenia: {model_name}\nDataset: {dataset} ({variant_desc})")
        plt.xlabel("Epoka")
        plt.ylabel("Strata (skala logarytmiczna)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(loc='upper right')

        # --- MODUŁ ZAPISU ---
        if save_name and dataset and folder:
            save_path = SystemPlotter.save(plt_=plt, save_name=save_name, dataset=dataset, folder=folder)
            print(f"    💾 Krzywa uczenia zapisana: {save_path}")

        plt.show(block=False)

        if not show:
            plt.close('all')

    @staticmethod
    def save(plt_, save_name=None, dataset=None, folder="Results"):
        full_path_dir = os.path.join(os.getcwd(), folder, dataset)
        os.makedirs(full_path_dir, exist_ok=True)

        # Plik ląduje bezpośrednio w folderze datasetu
        save_path = os.path.join(full_path_dir, f"{dataset}_{save_name}.png")

        plt_.savefig(save_path, dpi=300, bbox_inches='tight')

        return save_path


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