import numpy as np
import os


class DatasetHandler:
    """
        Klasa narzędziowa do zarządzania trwałym przechowywaniem danych systemowych.
        Obsługuje kompresowany zapis i odczyt macierzy NumPy, dbając o strukturę folderów:
        Projekt / Dataset / [Podfolder] / plik.npz.
    """

    # Pobranie ścieżki do folderu, w którym znajduje się ten skrypt (Dataset)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(CURRENT_DIR)

    # Domyślny podfolder dla zbiorów danych
    DEFAULT_SUBFOLDER = "Dataset1"

    @staticmethod
    def save(U, Y, DYDT, T, filename, folder=DEFAULT_SUBFOLDER):
        """
        Zapisuje macierze danych do skompresowanego pliku .npz.

        Automatycznie tworzy strukturę folderów, jeśli ta nie istnieje. Używa kompresji,
        aby zminimalizować rozmiar plików przy dużych zbiorach trajektorii.

        Args:
            U (np.ndarray): Macierz wymuszeń (sterowań).
            Y (np.ndarray): Macierz odpowiedzi układu (stanów).
            DYDT (np.ndarray): Macierz pochodnych stanów.
            T (np.ndarray): Wektor czasu (wspólny dla trajektorii).
            filename (str): Nazwa pliku wyjściowego (np. 'train_set.npz').
            folder (str): Nazwa podfolderu wewnątrz katalogu Dataset.
        """
        # Konstrukcja pełnej ścieżki docelowej
        full_folder_path = os.path.join(DatasetHandler.BASE_DIR, folder)

        # Tworzenie folderu, jeśli jest potrzebny
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        path = os.path.join(full_folder_path, filename)

        # Zapis z kompresją pod kluczami odpowiadającymi nazwom zmiennych
        np.savez_compressed(path, U=U, Y=Y, DYDT=DYDT, T=T)

        print(f"\n✅ Dane zapisane pomyślnie w: {path}")
        print(f"\tMetadane -> Trajektorie: {U.shape[0]}, Punkty: {U.shape[1]}")
        print(f"\tKształt U: {U.shape}, Y: {Y.shape}, DYDT: {DYDT.shape}, T: {T.shape}")

    @staticmethod
    def load(filename, folder=DEFAULT_SUBFOLDER):
        """
        Wczytuje zestaw danych z pliku .npz.

        Args:
            filename (str): Nazwa pliku do odczytu.
            folder (str): Podfolder, w którym znajduje się plik.

        Returns:
            tuple: (U, Y, DYDT, T) jako krotka macierzy NumPy.

        Raises:
            FileNotFoundError: Jeśli wskazany plik nie istnieje w podanej lokalizacji.
        """
        path = os.path.join(DatasetHandler.BASE_DIR, folder, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Nie znaleziono pliku bazy danych: {path}")

        # Odczyt danych z pliku npz
        data = np.load(path)
        U_all = data['U']
        Y_all = data['Y']
        DYDT_all = data['DYDT']
        T = data['T']

        print(f"📖 Wczytano zbiór danych: {path} | Liczba trajektorii: {len(U_all)}")
        return U_all, Y_all, DYDT_all, T

    @staticmethod
    def list_datasets(folder=DEFAULT_SUBFOLDER):
        """
        Przeszukuje wskazany folder w poszukiwaniu dostępnych plików baz danych (.npz).

        Args:
            folder (str): Podfolder do przeszukania wewnątrz katalogu Dataset.

        Returns:
            list[str]: Lista nazw znalezionych plików .npz.
        """
        full_path = os.path.join(DatasetHandler.BASE_DIR, folder)

        if not os.path.exists(full_path):
            print(f"⚠️ Folder '{folder}' nie istnieje.")
            return []

        files = [f for f in os.listdir(full_path) if f.endswith('.npz')]
        print(f"📂 Znalezione zbiory w '{folder}': {files}")
        return files
