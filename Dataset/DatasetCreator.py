import glob
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

    def __init__(self, obj, t_end=1000, dt=0.5, amp_range=(3, 9), noise_level=0.0):
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
        self.noise_level = noise_level

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
        self.generated_clean_signals = []     # Przechowuje kilka sygnałów do wizualizacji
        self.generated_noise_signals = []
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

    def create_dataset(self, n_trajectories, mode="train", folder="Dataset1"):
        """
            Generuje zadaną liczbę trajektorii i zapisuje je do skompresowanego pliku binarnego.
            Dodatkowo generuje raport JSON z parametrami sygnałów.

            Args:
                n_trajectories (int): Liczba trajektorii do wygenerowania.
                folder (str): Podfolder zapisu (wewnątrz katalogu Dataset).
                filename (str): Nazwa pliku wyjściowego (.npz).
        """
        # --- GENEROWANIE ŚCIEŻEK ---
        base_filename = f"{mode}_dataset_{n_trajectories}.npz"
        noise_filename = f"{mode}_dataset_{n_trajectories}_noise_{self.noise_level}.npz"

        # Ścieżka do sprawdzania (szukamy dowolnego pliku dla danego trybu w tym folderze)
        search_pattern = os.path.join("Dataset", folder, f"{mode}_dataset_*.npz")
        existing_files = glob.glob(search_pattern)

        # --- BEZPIECZNIK (ZABEZPIECZENIE) ---
        if existing_files:
            print("\n" + "!" * 90)
            print(f"⚠️  ALARM: Wykryto istniejące zbiory danych w folderze '{folder}'!")
            print("-" * 60)
            for f in existing_files:
                print(f"   • Znaleziono: {os.path.basename(f)}")

            print("\n🛑 INSTRUKCJA BEZPIECZEŃSTWA:")
            print("1. Musisz RĘCZNIE usunąć powyższe pliki (wszystkie z wybranego folderu), jeśli chcesz wygenerować nowe.")
            print("2. PAMIĘTAJ: Jeśli zmienisz dane, Twoje dotychczasowe modele MLP (Folder /ML/Saved_models)")
            print("   wytrenowane na tym folderze mogą stać się NIEAKTUALNE i należy je usunąć.")
            print("   Dodatkowo należy usunąć wykresy (Folder /Results/Plots), metryki (Folder /Results/Reports.")
            print("-" * 90)
            print("Przerwanie procesu w celu ochrony spójności danych...")
            print("!" * 90 + "\n")
            return 1 # Kończymy funkcję, nic nie generujemy

        U_list, Y_list, DYDT_list = [], [], []
        U_noise_list, Y_noise_list, DYDT_noise_list = [], [], []
        T = []

        self.generated_clean_signals = []  # Resetujemy listę podglądu
        self.generated_noise_signals = []  # Resetujemy listę podglądu

        self.history = []  # Reset historii
        self.stats = {'aprbs': 0, 'multisine': 0, 'noise': 0}

        print(f"\n--- Start generowania zestawu {n_trajectories} przebiegów: {base_filename} ---")
        if self.noise_level > 0.0:
            print(f"--- Dodatkowy zestaw {n_trajectories} przebiegów z poziomem szumu {self.noise_level}: {noise_filename} ---")

        # tqdm automatycznie stworzy pasek postępu
        for i in tqdm(range(n_trajectories), desc=f"Generowanie {mode}"):
            # 1. Losowanie i tworzenie sygnału sterującego
            u_func = self._get_random_signal()

            # Zapisujemy tylko pierwsze 10 sygnałów do ewentualnego podglądu (oszczędność RAM)
            if len(self.generated_clean_signals) < 10:
                self.generated_clean_signals.append(u_func)

            # Symulacja obiektu przy pomocy DataGeneratora
            gen = DataGenerator(obj=self.obj, u_func=u_func, dt=self.dt)
            system_data_clean = gen.generate()

            # Agregacja danych CZYSTYCH do list
            U_list.append(system_data_clean.u)
            Y_list.append(system_data_clean.y)
            DYDT_list.append(system_data_clean.dy_dt)

            # Agregacja danych ZASZUMIONYCH
            if self.noise_level > 0.0:
                system_noise_data = gen.add_noise(system_data_clean, noise_level=self.noise_level)

                # Sterowanie u i pochodne dy_dt zostają te same (szumimy tylko pomiar h)
                U_noise_list.append(system_noise_data.u)
                Y_noise_list.append(system_noise_data.y)  # Tutaj są zaszumione h1, h2
                DYDT_noise_list.append(system_noise_data.dy_dt)

            if i == 0:
                T = system_data_clean.t

        # Konwersja list na macierze numpy o kształcie (trajektorie, próbki, wymiar)
        U_clean_final = np.stack(U_list)
        Y_clean_final = np.stack(Y_list)
        DYDT_clean_final = np.stack(DYDT_list)

        # Zapis binarny za pomocą DatasetHandler
        DatasetHandler.save(U_clean_final, Y_clean_final, DYDT_clean_final, T, base_filename, folder)

        if self.noise_level > 0.0:
            U_noise_final = np.stack(U_noise_list)
            Y_noise_final = np.stack(Y_noise_list)
            DYDT_noise_final = np.stack(DYDT_noise_list)

            DatasetHandler.save(U_noise_final, Y_noise_final, DYDT_noise_final, T, noise_filename, folder)

        print(f"\tStatystyki: {self.stats}")

        # Zapis raportu JSON
        self._save_report(base_filename, folder, U_clean_final.shape, Y_clean_final.shape, DYDT_clean_final.shape)
        # print("-" * 60)

        return 0

    def show_random_signals(self, n=3):
        """
        Wizualizuje przykładowe sygnały sterujące wygenerowane w ostatniej sesji.

        Args:
            n (int): Liczba losowych sygnałów do wyświetlenia.
        """

        if not self.generated_clean_signals:
            print("\n❌ Brak sygnałów do wyświetlenia. Uruchom najpierw create_dataset.")
            return

        n = min(n, len(self.generated_clean_signals))
        indices = np.random.choice(len(self.generated_clean_signals), n, replace=False)

        plt.figure(figsize=(12, 3 * n))
        for i, idx in enumerate(indices):
            sig = self.generated_clean_signals[idx]
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
                "amp_tange": self.amp_range,
                "noise_level": self.noise_level
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
