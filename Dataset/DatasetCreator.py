import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import os
from matplotlib import pyplot as plt
from Dataset.DatasetHandler import DatasetHandler
from Generators.DataGenerator import DataGenerator
from Generators.SignalGenerator import APRBSSignal, MultisineSignal, FilteredNoiseSignal


class DatasetCreator:
    """
    Klasa zarządzająca masową generacją danych treningowych, walidacyjnych i testowych.
    Automatyzuje proces tworzenia zróżnicowanych zestawów danych dla sieci MLP,
    łącząc różne typy wymuszeń (APRBS, Multisine, Noise) w jeden zbiór.
    """

    def __init__(self, obj, t_end=1000, dt=0.5, amp_range=(3, 9)):
        """
        Inicjalizacja kreatora zbioru danych.

        Args:
            obj: Obiekt fizyczny (model matematyczny) systemu do symulacji.
            t_end (int): Czas trwania każdej trajektorii w sekundach.
            dt (float): Krok próbkowania symulacji.
            amp_range (tuple): Zakres dopuszczalnych amplitud sygnału sterującego (min, max).
        """

        self.obj = obj
        self.t_end = t_end
        self.dt = dt
        self.amp_range = amp_range

        # --- KONFIGURACJA ZAKRESÓW LOSOWANIA PARAMETRÓW SYGNAŁÓW ---
        self.config = {
            "aprbs": {
                "h_min_range": (50, 150),   # Minimalny czas trwania kroku
                "h_add_range": (100, 300)   # Dodatkowy czas trwania kroku (losowy)
            },
            "multisine": {
                "n_freqs_range": (5, 15),   # Liczba składowych harmonicznych
                "f_max_range": (0.005, 0.02) # Maksymalna częstotliwość sygnału
            },
            "noise": {
                "cutoff_range": (0.001, 0.01) # Częstotliwość odcięcia filtru dolnoprzepustowego
            }
        }

        # --- STATYSTYKI I HISTORIA ---
        self.stats = {'aprbs': 0, 'multisine': 0, 'noise': 0}
        self.generated_signals = []     # Przechowuje kilka sygnałów do wizualizacji
        self.history = []               # Log parametrów każdego wygenerowanego przebiegu

    def _get_random_signal(self):
        """
        Losuje typ sygnału i generuje go z losowymi parametrami z zadanego zakresu.

        Returns:
            Signal: Obiekt jednej z klas generatorów sygnałów (APRBS, Multisine lub Noise).
        """

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
        """
            Generuje zadaną liczbę trajektorii i zapisuje je do skompresowanego pliku binarnego.
            Dodatkowo generuje raport JSON z parametrami sygnałów.

            Args:
                n_trajectories (int): Liczba trajektorii do wygenerowania.
                folder (str): Podfolder zapisu (wewnątrz katalogu Dataset).
                filename (str): Nazwa pliku wyjściowego (.npz).
        """

        U_list, Y_list, DYDT_list, T = [], [], [], []
        self.generated_signals = []  # Resetujemy listę podglądu
        self.history = []  # Reset historii
        self.stats = {'aprbs': 0, 'multisine': 0, 'noise': 0}

        print(f"\n--- Start generowania zestawu {n_trajectories} przebiegów: {filename} ---")

        # tqdm automatycznie stworzy pasek postępu
        for i in tqdm(range(n_trajectories), desc="Postęp generowania"):
            # 1. Losowanie i tworzenie sygnału sterującego
            u_func = self._get_random_signal()

            # Zapisujemy tylko pierwsze 10 sygnałów do ewentualnego podglądu (oszczędność RAM)
            if len(self.generated_signals) < 10:
                self.generated_signals.append(u_func)

            # 2. Symulacja obiektu przy pomocy DataGeneratora
            gen = DataGenerator(self.obj, u_func, self.dt)
            system_data = gen.generate()

            # 3. Agregacja danych do list
            U_list.append(system_data.u)
            Y_list.append(system_data.y)
            DYDT_list.append(system_data.dy_dt)

            if i == 0:
                T = system_data.t

        # Konwersja list na macierze numpy o kształcie (trajektorie, próbki, wymiar)
        U_final = np.stack(U_list)
        Y_final = np.stack(Y_list)
        DYDT_final = np.stack(DYDT_list)

        # Zapis binarny za pomocą DatasetHandler
        DatasetHandler.save(U_final, Y_final, DYDT_final, T, filename, folder)

        print(f"\tStatystyki: {self.stats}")

        # Zapis raportu JSON
        self._save_report(filename, folder, U_final.shape, Y_final.shape, DYDT_final.shape)
        # print("-" * 60)

    def show_random_signals(self, n=3):
        """
        Wizualizuje przykładowe sygnały sterujące wygenerowane w ostatniej sesji.

        Args:
            n (int): Liczba losowych sygnałów do wyświetlenia.
        """

        if not self.generated_signals:
            print("\n❌ Brak sygnałów do wyświetlenia. Uruchom najpierw create_dataset.")
            return

        n = min(n, len(self.generated_signals))
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
        plt.title("Przykładowe sygnały sterujące z wygenerowanego zbioru", y=1.02)
        plt.show()

    def _save_report(self, filename, folder, u_shape, y_shape, dydt_shape):
        """
        Zapisuje szczegółowy raport techniczny o zbiorze danych do pliku JSON.
        Raport zawiera metadane, wymiary macierzy oraz historię parametrów losowania.

        Args:
            filename (str): Nazwa pliku .npz, dla którego tworzony jest raport.
            folder (str): Folder zapisu.
            u_shape (tuple): Kształt macierzy sterowań.
            y_shape (tuple): Kształt macierzy wyjść.
            dydt_shape (tuple): Kształt macierzy pochodnych.
        """

        report = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "system_object": str(self.obj.__class__.__name__),
                "dt": self.dt,
                "t_end": self.t_end,
                "amp_tange": self.amp_range
            },
            "dataset_structure": {
                "total_trajectories": u_shape[0],
                "points_per_trajectory": {
                    "raw_u_y": u_shape[1],  # N punktów
                    "derivative_dydt": dydt_shape[1]  # N-1 punktów
                },
                "dimensions": {
                    "u_control_dim": u_shape[2] if len(u_shape) > 2 else 1,
                    "y_state_dim": y_shape[2] if len(y_shape) > 2 else 1,
                    "dydt_dim": dydt_shape[2] if len(dydt_shape) > 2 else 1
                }
            },
            "statistics": self.stats,
            "signals_config": self.config,
            "run_history": self.history
        }

        report_name = filename.replace(".npz", "_info.json")
        report_path = os.path.join(DatasetHandler.BASE_DIR, folder, report_name)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"\n📄 Raport techniczny zapisany: {report_path}")
