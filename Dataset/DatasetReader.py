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


