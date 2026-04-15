import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from Dataset.DatasetHandler import DatasetHandler
from Generators.SystemData import SystemData


class DatasetReader:
    """
        Klasa odpowiedzialna za odczyt zestawów danych i ich wizualizację.
        Działa jako interfejs między zapisanymi plikami binarnymi .npz a obiektami klasy SystemData.
    """

    def __init__(self):
        """
            Inicjalizuje czytnik danych z pustą listą trajektorii.
        """
        self.data_list = []
        self.folder = None
        self.filename = None

    def find_and_read(self, folder, mode, noise_level=0.0):
        """
        folder: np. 'Dataset5'
        mode: np. 'train'
        noise_level: 0.2 lub 0.0
        """
        import glob
        import os

        # Budujemy bazowy wzorzec wyszukiwania
        if noise_level > 0.0:
            # Formatowanie %g usuwa zbędne zera po przecinku (np. 1.0 -> 1, 0.2 -> 0.2)
            noise_str = "%g" % noise_level
            pattern = f"{mode}_dataset_*_noise_{noise_str}.npz"
        else:
            # Szukamy czystych plików
            pattern = f"{mode}_dataset_*.npz"

        search_path = os.path.join("Dataset", folder, pattern)
        all_matches = glob.glob(search_path)

        # --- KRYTYCZNY FILTR ---
        if noise_level == 0.0:
            # Jeśli chcemy CLEAN, bierzemy tylko pliki BEZ słowa 'noise' w nazwie
            matches = [f for f in all_matches if "_noise_" not in os.path.basename(f)]
        else:
            matches = all_matches

        if len(matches) == 0:
            raise FileNotFoundError(f"❌ Nie znaleziono pliku dla {mode} (noise={noise_level}) w {folder}")

        if len(matches) > 1:
            raise RuntimeError(f"⚠️ Za dużo plików pasuje do wzorca w {folder} dla {mode}!")

        filename = os.path.basename(matches[0])
        print(f"📂 Wczytuję: {filename}")
        return self.read_all(folder=folder, filename=filename)

    def read_all(self, folder="Dataset1", filename="train_set.npz"):
        """
        Odtwarza listę obiektów SystemData z zapisanego pliku .npz.

        Metoda ładuje macierze sterowań, wyjść i pochodnych, a następnie
        paczkuje je w listę obiektów klasy SystemData dla łatwiejszego zarządzania.

        Args:
            folder (str): Nazwa folderu, w którym znajduje się plik.
            filename (str): Nazwa pliku .npz do wczytania.

        Returns:
            list[SystemData]: Lista wczytanych obiektów SystemData reprezentujących trajektorie.
        """

        self.folder = folder
        self.filename = filename

        # Załadowanie surowych macierzy z pliku przy pomocy handlera
        U_all, Y_all, DYDT_all, T_single = DatasetHandler.load(filename, folder)

        self.data_list = []
        # Iteracja po pierwszej osi (liczba trajektorii) i tworzenie obiektów SystemData
        for i in range(U_all.shape[0]):
            sd = SystemData(y=Y_all[i], u=U_all[i], t=T_single, dydt=DYDT_all[i])
            self.data_list.append(sd)

        # print(f"📖 Wczytano {len(self.data_list)} przebiegów z pliku {filename}")
        return self.data_list

    def show_random_signals(self, idx=None):
        """
        Wizualizuje wybraną lub losową trajektorię z wczytanego zestawu.

        Metoda automatycznie pobiera dane przygotowane do wykresu (wyrównane czasowo),
        a następnie wykorzystuje SystemPlotter do wygenerowania przebiegów.

        Args:
            idx (int, optional): Indeks trajektorii do wyświetlenia.
                                Jeśli None, zostanie wybrany losowy indeks.

        Note:
            Metoda wymaga wcześniejszego wywołania read_all().
        """
        if not self.data_list:
            print("❌ Brak danych w pamięci. Najpierw wywołaj read_all().")
            return

        # 1. Wybór indeksu trajektorii
        if idx is None:
            idx = np.random.randint(len(self.data_list))

        # Pobranie obiektu SystemData z listy
        obj = self.data_list[idx]

        # 2. Pobranie danych przygotowanych pod wykres (wyrównanie wymiarów N-2)
        t_plot, u_plot, h_true, dh_dt_true = obj.get_data_to_plot()

        # 3. Dynamiczny import Plottera w celu uniknięcia cyklicznych zależności
        from Utills.SystemPlotter import SystemPlotter

        # 4. Generowanie wykresu przy użyciu ujednoliconego narzędzia wizualizacji
        SystemPlotter.plot(
            t=t_plot,
            u=u_plot,
            y_true=h_true,
            dy_dt_true=dh_dt_true,
            title=f"Podgląd surowych danych (Trajektoria {idx}) z pliku {self.folder}/{self.filename}"
        )

        plt.show(block=True)

    def check_available_variants(self, folder):
        """
        Skanuje folder i sprawdza, czy są komplety plików (train, val, test).
        Zwraca listę słowników z dostępnymi wariantami.
        """
        variants = []
        path = os.path.join(os.getcwd(), "Dataset", folder)

        if not os.path.exists(path):
            return variants

        all_files = os.listdir(path)

        # 1. Sprawdzamy wersję CLEAN (pliki .npz bez słowa 'noise')
        clean_files = [f for f in all_files if f.endswith('.npz') and 'noise' not in f]
        has_clean = all(any(f.startswith(mode) for f in clean_files) for mode in ['train', 'val', 'test'])

        if has_clean:
            variants.append({"name": "CLEAN", "noise": 0.0})

        # 2. Sprawdzamy wersję NOISY
        # Szukamy unikalnych poziomów szumu w nazwach plików
        noise_levels = set()
        for f in all_files:
            if '_noise_' in f and f.endswith('.npz'):
                # Wyciągamy wartość szumu z nazwy (np. '0.05' z '...noise_0.05.npz')
                parts = f.replace('.npz', '').split('_noise_')
                if len(parts) > 1:
                    noise_levels.add(parts[1])

        for nl in noise_levels:
            # Sprawdzamy czy dla tego konkretnego szumu mamy komplet 3 plików
            noisy_files = [f for f in all_files if f"noise_{nl}" in f]
            has_noisy_set = all(any(f.startswith(mode) for f in noisy_files) for mode in ['train', 'val', 'test'])

            if has_noisy_set:
                variants.append({"name": f"NOISY_{nl}", "noise": float(nl)})

        return variants

