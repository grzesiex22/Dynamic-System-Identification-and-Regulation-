import os

import numpy as np
import pandas as pd


class Metrics:
    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(Metrics.mse(y_true, y_pred)))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def max_error(y_true, y_pred):
        """Oblicza największy błąd bezwzględny (błąd amplitudy)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Obliczamy maksimum z wartości bezwzględnych różnic
        return float(np.max(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "MSE": Metrics.mse(y_true, y_pred),
            "RMSE": Metrics.rmse(y_true, y_pred),
            "MAE": Metrics.mae(y_true, y_pred),
            "MAX_ERR": Metrics.max_error(y_true, y_pred),
            "R2": Metrics.r2(y_true, y_pred),
        }

    def print_metrics(title, metrics):
        print(f"\n=== {title} ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.8f}")


class MetricsSummarizer:
    """
    Klasa do zbierania, porównywania i wyświetlania metryk z wielu modeli.
    """

    def __init__(self):
        # Słownik przechowujący metryki: { 'Nazwa Modelu': { 'MSE': 0.1, ... } }
        self.results = {}

    def add_metrics(self, traj_idx, model_name, metrics_dict):
        """
            Dodaje metryki dla konkretnego modelu.
        """
        key = (f"Trajektoria {traj_idx}", model_name)
        self.results[key] = metrics_dict

    def get_best_model(self, metric_name="MSE"):
        """
        Zwraca (Trajektoria, Model) oraz wartość dla najlepszego wyniku.
        """
        if not self.results:
            return None, None

        # Szukamy klucza (Traj, Model), dla którego wartość metryki jest najniższa
        best_key = min(
            self.results,
            key=lambda k: self.results[k].get(metric_name, float('inf'))
        )

        traj_info, model_info = best_key
        value = self.results[best_key][metric_name]

        return (traj_info, model_info), value

    def get_overall_average(self):
        """
        Oblicza średnie metryki dla każdego modelu ze wszystkich trajektorii.
        Zwraca DataFrame ze średnimi.
        """
        if not self.results:
            return None

        # 1. Tworzymy DF i rozbijamy MultiIndex na kolumny
        df = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        df.columns = ['Traj', 'Model'] + list(df.columns[2:])

        # 2. Grupowanie po modelu i liczenie średniej
        # numeric_only=True pominie kolumnę 'Traj' automatycznie
        avg_df = df.groupby('Model').mean(numeric_only=True)

        return avg_df

    def save_all_to_file(self, dataset="Dataset1", folder="Results"):
        # Tworzymy ścieżkę: Results/Dataset1
        full_path_dir = os.path.join(os.getcwd(), folder, dataset)
        os.makedirs(full_path_dir, exist_ok=True)

        # Plik ląduje bezpośrednio w folderze datasetu
        path = os.path.join(full_path_dir, f"{dataset}_Test_results.csv")

        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.to_csv(path)
        print(f"💾 Metryki zapisane w: {path}")

    def save_averages_to_file(self, dataset="Dataset1", folder="Results"):
        """
        Zapisuje uśrednione wyniki oraz całkowite czasy pracy modeli do pliku CSV.
        """
        avg_df = self.get_overall_average()
        if avg_df is None:
            return

        # 1. Obliczamy sumaryczne czasy (tak samo jak w show_averages)
        df_temp = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        df_temp.columns = ['Traj', 'Model'] + list(df_temp.columns[2:])

        # Grupowanie i sumowanie czasów
        total_times = df_temp.groupby('Model')[['Time [s]', 'Time [min]']].sum()
        # Zmiana nazw kolumn, żeby w CSV było jasne, że to sumy
        total_times.columns = ['Total Time [s]', 'Total Time [min]']

        # 2. Łączymy średnie metryki z sumarycznymi czasami
        # Oba DataFrame'y mają 'Model' jako indeks, więc zadziała proste join/concat
        final_df = pd.concat([avg_df, total_times], axis=1)

        # 3. Przygotowanie ścieżki i zapis
        target_dir = os.path.join(os.getcwd(), folder, dataset)
        os.makedirs(target_dir, exist_ok=True)

        path = os.path.join(target_dir, f"{dataset}_Test_avg_results.csv")

        final_df.to_csv(path)
        print(f"🏆 Średnie wyniki i czasy całkowite zapisane w: {path}")

    def show_averages(self):
        """
        Pobiera średnie i printuje je w sformatowanej tabeli.
        """
        avg_df = self.get_overall_average()
        if avg_df is None:
            print("Brak danych do uśrednienia.")
            return

        # Obliczamy sumę czasów (ogólny czas pracy modelu)
        # Tworzymy roboczy DF, żeby łatwo wyciągnć sumę
        df_temp = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        df_temp.columns = ['Traj', 'Model'] + list(df_temp.columns[2:])
        total_times_s = df_temp.groupby('Model')['Time [s]'].sum()
        total_times_min = df_temp.groupby('Model')['Time [min]'].sum()

        # KONFIGURACJA SZEROKOŚCI
        first_col_w = 20  # Dla kolumny "Model"
        rest_col_w = 16  # Dla wszystkich metryk i czasów
        # Nagłówki
        metrics_headers = list(avg_df.columns)
        time_headers = ["Total Time [s]", "Total Time [min]"]

        # Budowanie nagłówka (string)
        header_str = f"{'Model':^{first_col_w}} | "
        header_str += " | ".join(f"{h:^{rest_col_w}}" for h in metrics_headers + time_headers)

        total_w = len(header_str) + 4
        line = "═" * total_w

        print("\n" + line)
        print(f"║{'📊 ŚREDNIE WYNIKI ZBIORCZE'.center(total_w - 3)}║")
        print(line)
        print(f"| {header_str} |")
        print("-" * total_w)

        for model_name, row in avg_df.iterrows():
            # 1. Nazwa modelu (pierwsza kolumna)
            row_str = f"{model_name:<{first_col_w}} | "

            # 2. Metryki (średnie)
            row_str += " | ".join(f"{v:^{rest_col_w}.8f}" for v in row)

            # 3. Czasy (sumy)
            t_s = total_times_s[model_name]
            t_m = total_times_min[model_name]
            row_str += f" | {t_s:^{rest_col_w}.4f} | {t_m:^{rest_col_w}.4f}"

            print(f"| {row_str} |")

        print(line + "\n")

    def show_all(self):
        """
        Wyświetla szczegółowe porównanie dla każdej trajektorii.
        """
        if not self.results:
            print("Brak danych do wyświetlenia.")
            return None

        # 1. Tworzymy DataFrame
        df = pd.DataFrame.from_dict(self.results, orient='index')

        # 2. Ustawienia szerokości
        first_col_w = 20  # Dla "Trajektoria"
        rest_col_w = 16  # Dla reszty kolumn (Model, Metryki, Czas)

        # Przygotowanie nagłówków
        # df.columns zawiera już "Time [s]" i "Time [min]", bo dodaliśmy je w run()
        columns = ["Trajektoria", "Model"] + list(df.columns)

        header_str = f"{columns[0]:^{first_col_w}} | {columns[1]:^{rest_col_w}}"
        for col in columns[2:]:
            header_str += f" | {col:^{rest_col_w}}"

        total_width = len(header_str) + 4
        thick_line = "═" * total_width
        thin_line = "─" * total_width  # Zmieniona na cieńszą dla lepszej czytelności

        # 3. Nagłówek tabeli
        print("\n" + thick_line)
        print(f"║{'📊 SZCZEGÓŁOWE PORÓWNANIE METRYK'.center(total_width - 3)}║")
        print(thick_line)
        print(f"| {header_str} |")
        print(thick_line.replace("═", "─"))

        # 4. Dane
        last_traj = None
        for (traj, model), row in df.iterrows():
            # Linia oddzielająca grupy trajektorii
            if last_traj is not None and traj != last_traj:
                print(thin_line)

            # Pierwsza kolumna (Trajektoria)
            row_str = f"{str(traj):<{first_col_w}} | "
            # Druga kolumna (Model)
            row_str += f"{model:<{rest_col_w}}"

            # Reszta kolumn (Metryki i Czas konkretnej symulacji)
            for val in row:
                # Sprawdzamy czy to czas (mniej miejsc po przecinku) czy metryka
                if val < 0.0001:  # Bardzo małe błędy MSE
                    row_str += f" | {val:^{rest_col_w}.8e}"  # Notacja naukowa dla precyzji
                else:
                    row_str += f" | {val:^{rest_col_w}.8f}"

            print(f"| {row_str} |")
            last_traj = traj

        print(thick_line + "\n")
        return df
