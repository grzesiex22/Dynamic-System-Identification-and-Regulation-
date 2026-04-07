import json
from datetime import datetime
from tqdm import tqdm
import time
import numpy as np
import os
from matplotlib import pyplot as plt
from Dataset.DatasetHandler import DatasetHandler
from Generators.DataGenerator import DataGenerator
from Generators.SignalGenerator import APRBSSignal, MultisineSignal, FilteredNoiseSignal


class DatasetManager:
    """
    Klasa zarządzająca masową generacją danych treningowych, walidacyjnych i testowych.
    Automatyzuje proces tworzenia zróżnicowanych zestawów danych dla sieci MLP.
    """

    def __init__(self, obj, t_end=1000, dt=0.5, amp_range=(3, 9)):
        self.obj = obj
        self.t_end = t_end
        self.dt = dt
        self.amp_range = amp_range

        # --- KONFIGURACJA ZAKRESÓW LOSOWANIA ---
        self.config = {
            "aprbs": {
                "h_min_range": (50, 150),
                "h_add_range": (100, 300)
            },
            "multisine": {
                "n_freqs_range": (5, 15),
                "f_max_range": (0.005, 0.02)
            },
            "noise": {
                "cutoff_range": (0.001, 0.01)
            }
        }

        # --- STATYSTYKI ---
        self.stats = {'aprbs': 0, 'multisine': 0, 'noise': 0}
        self.generated_signals = []
        self.history = []  # Tutaj zapiszemy parametry każdego przebiegu

    def _get_random_signal(self):
        """Losuje sygnał i zapamiętuje jego dokładne parametry."""
        choice = np.random.choice(['aprbs', 'multisine', 'noise'])
        self.stats[choice] += 1

        run_params = {"type": choice}

        if choice == 'aprbs':
            c = self.config["aprbs"]
            h_min = np.random.randint(*c["h_min_range"])
            h_max = h_min + np.random.randint(*c["h_add_range"])

            run_params.update({"h_min": h_min, "h_max": h_max})
            sig = APRBSSignal(self.t_end, self.dt, self.amp_range, (h_min, h_max))

        elif choice == 'multisine':
            c = self.config["multisine"]
            n_f = np.random.randint(*c["n_freqs_range"])
            f_m = np.random.uniform(*c["f_max_range"])

            run_params.update({"n_freqs": n_f, "f_max": f_m})
            sig = MultisineSignal(self.t_end, self.dt, self.amp_range, n_f, f_m)

        else:  # noise
            c = self.config["noise"]
            cut = np.random.uniform(*c["cutoff_range"])

            run_params.update({"cutoff": cut})
            sig = FilteredNoiseSignal(self.t_end, self.dt, self.amp_range, cut)

        self.history.append(run_params)
        return sig

    def create_dataset(self, n_trajectories, folder="Dataset1", filename="train_set.npz"):
        """Generuje N trajektorii i zapisuje je do skompresowanego pliku."""

        X_list, Y_list, T_list = [], [], []
        self.generated_signals = []  # Resetujemy listę podglądu
        self.history = []  # Reset historii
        self.stats = {'aprbs': 0, 'multisine': 0, 'noise': 0}

        print(f"--- Start generowania zestawu {n_trajectories} przebiegów: {filename} ---")

        # tqdm automatycznie stworzy pasek postępu
        for i in tqdm(range(n_trajectories), desc="Postęp generowania"):
            # 1. Losuj sygnał
            u_func = self._get_random_signal()

            # Zapisujemy tylko pierwsze 10 sygnałów do ewentualnego podglądu (oszczędność RAM)
            if len(self.generated_signals) < 10:
                self.generated_signals.append(u_func)

            # 2. Symuluj
            gen = DataGenerator(self.obj, u_func, self.dt)
            system_data = gen.generate()

            # 3. Wyciągnij dane treningowe (pary [u, y] -> dy/dt)
            X, Y, _ = system_data.get_training_data()

            X_list.append(X)
            Y_list.append(Y)

            if i == 0:
                _, _, T = system_data.get_training_data()
                T_list.append(T)

        X_final = np.stack(X_list)  # Kształt: (n_trajectories, n_samples, 3)
        Y_final = np.stack(Y_list)  # Kształt: (n_trajectories, n_samples, 2)
        T_final = np.stack(T_list)  # Kształt: (n_trajectories, n_samples, 2)

        print(f"Zakończono! Stworzono łącznie {X_final.shape} próbek.")
        print(f"Statystyki sygnałów: {self.stats}")
        print("-" * 40)

        DatasetHandler.save(X_final, Y_final, T_final, filename, folder)

        # Zapis raportu JSON
        self._save_report(filename, folder, X_final.shape, Y_final.shape)

        print(f"Zakończono! Statystyki: {self.stats}")

    def show_random_signals(self, n=3):
        """Wyświetla n przykładowych sygnałów sterujących z ostatniej sesji."""
        if not self.generated_signals:
            print("Brak sygnałów do wyświetlenia. Uruchom najpierw create_dataset.")
            return

        n = min(n, len(self.generated_signals))
        # Losujemy indeksy sygnałów do pokazania
        indices = np.random.choice(len(self.generated_signals), n, replace=False)

        plt.figure(figsize=(12, 3 * n))
        for i, idx in enumerate(indices):
            sig = self.generated_signals[idx]
            plt.subplot(n, 1, i + 1)
            plt.plot(sig.t, sig.generate(), label=f"Typ: {type(sig).__name__}")
            plt.ylabel("Napięcie [V]")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')

        plt.xlabel("Czas [s]")
        plt.tight_layout()
        plt.suptitle("Przykładowe sygnały sterujące z wygenerowanego zbioru", y=1.02)
        plt.show()

    def _save_report(self, filename, folder, x_shape, y_shape):
        report = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "total_trajectories": x_shape[0],
                "samples_per_trajectory": x_shape[1],
                "input_dim": x_shape[2],  # Liczba cech wejściowych (u, h1, h2)
                "output_dim": y_shape[2],  # Liczba celów (np. dy1/dt, dy2/dt)
                "dt": self.dt,
                "t_end": self.t_end,
                "system_object": str(self.obj.__class__.__name__)
            },
            "statistics": self.stats,
            "configuration": self.config,
            "trajectory_details": self.history
        }

        report_name = filename.replace(".npz", "_info.json")
        report_path = os.path.join(DatasetHandler.BASE_DIR, folder, report_name)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"📄 Raport zaktualizowany o dane Y: {report_path}")