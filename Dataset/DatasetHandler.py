import numpy as np
import os


class DatasetHandler:
    """
        Klasa obsługująca zapis i odczyt danych z uwzględnieniem struktury:
        Projekt / Dataset / Dataset1 / plik.npz
    """

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(CURRENT_DIR)  # To już jest folder "Dataset"

    DEFAULT_SUBFOLDER = "Dataset1"

    @staticmethod
    def save(X, Y, T, filename, folder=DEFAULT_SUBFOLDER):
        """Zapisuje macierze do Dataset/folder/filename.npz"""
        # Łączymy ścieżkę: Dataset/Dataset1
        full_folder_path = os.path.join(DatasetHandler.BASE_DIR, folder)

        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        path = os.path.join(full_folder_path, filename)
        np.savez_compressed(path, X=X, Y=Y, T=T)

        print(f"✅ Dane zapisane w: {path}")
        print(f"   Kształt X: {len(X)}, Y: {len(Y)}, T: {len(T)}")

    @staticmethod
    def load(filename, folder=DEFAULT_SUBFOLDER):
        """Wczytuje dane z Dataset/folder/filename.npz"""
        path = os.path.join(DatasetHandler.BASE_DIR, folder, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Nie znaleziono pliku: {path}")

        data = np.load(path)
        X_all, Y_all, T = data['X'], data['Y'], data['T']

        print(f"📖 Wczytano: {path} | Próbek: {len(X_all)}")
        return X_all, Y_all, T

    @staticmethod
    def list_datasets(folder="data"):
        """Pomocnicza funkcja do sprawdzania co mamy w folderze"""
        if not os.path.exists(folder):
            return []
        files = [f for f in os.listdir(folder) if f.endswith('.npz')]
        print(f"📂 Dostępne zbiory w '{folder}': {files}")
        return files
