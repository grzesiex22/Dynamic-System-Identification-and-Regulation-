import os

import numpy as np
import pandas as pd


class Metrics:
    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # axis=0 sprawia, że liczymy średnią po wierszach, zostają 2 kolumny
        return np.mean((y_true - y_pred) ** 2, axis=0)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(Metrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred), axis=0)

    @staticmethod
    def max_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.max(np.abs(y_true - y_pred), axis=0)

    @staticmethod
    def r2(y_true, y_pred):
        # y_true = np.asarray(y_true)
        # y_pred = np.asarray(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

        # Obsługa dzielenia przez zero, jeśli dane są stałe
        res = np.zeros_like(ss_res)
        mask = ss_tot != 0
        res[mask] = 1.0 - (ss_res[mask] / ss_tot[mask])

        return res

    @staticmethod
    def iae(y_true, y_pred, dt=1.0):
        """
        Integral Absolute Error (IAE).
        IAE = sum(|y_true - y_pred|) * dt
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Obliczamy sumę modułów różnic i mnożymy przez krok czasu
        # axis=0 liczy sumę dla każdego sygnału wyjściowego osobno
        return np.sum(np.abs(y_true - y_pred), axis=0) * dt

    @staticmethod
    def evaluate(y_true, y_pred, t, suffixes=None):
        """
        Zwraca słownik z metrykami rozbitymi na Zbiornik 1 (Y1) i Zbiornik 2 (Y2).
        Zakłada, że Y ma kształt (N, 2).
        """

        # Jeśli nie podano sufiksów, generujemy domyślne (Y1, Y2, ...)
        num_signals = y_true.shape[1] if len(y_true.shape) > 1 else 1
        if suffixes is None:
            suffixes = [f"_Y{i + 1}" for i in range(num_signals)]

        if len(suffixes) != num_signals:
            raise ValueError(f"Liczba sufiksów ({len(suffixes)}) musi odpowiadać liczbie sygnałów ({num_signals})")

        # Obliczamy wektory metryk
        mse_v = Metrics.mse(y_true, y_pred)
        rmse_v = Metrics.rmse(y_true, y_pred)
        mae_v = Metrics.mae(y_true, y_pred)
        max_v = Metrics.max_error(y_true, y_pred)
        r2_v = Metrics.r2(y_true, y_pred)

        dt = t[1] - t[0]
        iae_v = Metrics.iae(y_true, y_pred, dt=dt)

        # Budujemy słownik dynamicznie
        results = {}
        for i, sfx in enumerate(suffixes):
            results[f"MSE{sfx}"] = float(mse_v[i])
            results[f"RMSE{sfx}"] = float(rmse_v[i])
            results[f"MAE{sfx}"] = float(mae_v[i])
            results[f"MAX_ERR{sfx}"] = float(max_v[i])
            results[f"IAE{sfx}"] = float(iae_v[i])
            results[f"R2{sfx}"] = float(r2_v[i])

        # Dodajemy metryki uśrednione (ogólne)
        results["R2_AVG"] = float(np.mean(r2_v))
        results["MSE_AVG"] = float(np.mean(mse_v))

        return results

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

    def add_metrics(self, traj_idx, model_name, dataset_kind, metrics_dict):
        """
            Dodaje metryki dla konkretnego modelu.
        """
        key = (f"Trajektoria {traj_idx}", model_name, dataset_kind)
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
        df.columns = ['Traj', 'Model', "Dataset_kind"] + list(df.columns[3:])

        # 2. Grupowanie po modelu i liczenie średniej
        # numeric_only=True pominie kolumnę 'Traj' automatycznie
        avg_df = df.groupby(['Model', 'Dataset_kind']).mean(numeric_only=True)

        return avg_df

    def save_all_to_file(self, dataset="Dataset1", folder="Results", save_name_sufix=None):
        # Tworzymy ścieżkę: Results/Dataset1
        full_path_dir = os.path.join(os.getcwd(), folder, dataset)
        os.makedirs(full_path_dir, exist_ok=True)

        # Plik ląduje bezpośrednio w folderze datasetu
        if save_name_sufix is None:
            filename = f"{dataset}_Test_results.csv"
        else:
            filename = f"{dataset}_Test_results{save_name_sufix}.csv"

        path = os.path.join(full_path_dir, filename)

        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.index.names = ['Traj', 'Model', 'Dataset_kind']

        df.to_csv(path)
        print(f"💾 Metryki zapisane w: {path}")

    def save_averages_to_file(self, dataset="Dataset1", folder="Results", save_name_sufix=None):
        """
        Zapisuje uśrednione wyniki oraz całkowite czasy pracy modeli do pliku CSV.
        """
        avg_df = self.get_overall_average()
        if avg_df is None:
            print("Brak danych (funkcja save_averages_to_file()")
            return

        # 1. Obliczamy sumaryczne czasy (tak samo jak w show_averages)
        df_temp = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        df_temp.columns = ['Traj', 'Model', 'Dataset_kind'] + list(df_temp.columns[3:])

        # Grupowanie i sumowanie czasów
        total_times = df_temp.groupby(['Model', 'Dataset_kind'])[['Time [s]', 'Time [min]']].sum()
        # Zmiana nazw kolumn, żeby w CSV było jasne, że to sumy
        total_times.columns = ['Total Time [s]', 'Total Time [min]']

        # 2. Łączymy średnie metryki z sumarycznymi czasami
        # avg_df po get_overall_average() ma Model i Dataset_kind jako indeks (lub kolumny)
        final_df = pd.concat([avg_df, total_times], axis=1)

        # 3. Przygotowanie ścieżki i zapis
        full_path_dir = os.path.join(os.getcwd(), folder, dataset)
        os.makedirs(full_path_dir, exist_ok=True)

        if save_name_sufix is None:
            filename = f"{dataset}_Test_avg_results.csv"
        else:
            filename = f"{dataset}_Test_avg_results{save_name_sufix}.csv"

        path = os.path.join(full_path_dir, filename)

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

        # Upewniamy się, że avg_df ma ustawiony MultiIndex do łatwego wyciągania czasów
        if not isinstance(avg_df.index, pd.MultiIndex):
            avg_df = avg_df.set_index(['Model', 'Dataset_kind'])

        # Obliczamy sumy czasów
        df_temp = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        df_temp.columns = ['Traj', 'Model', 'Dataset_kind'] + list(df_temp.columns[3:])

        times = df_temp.groupby(['Model', 'Dataset_kind'])[['Time [s]', 'Time [min]']].sum()

        # KONFIGURACJA SZEROKOŚCI
        model_col_w = 29
        kind_col_w = 10
        rest_col_w = 14

        metrics_headers = list(avg_df.columns)
        time_headers = ["Total Time [s]", "Total Time [min]"]

        # Nagłówek tabeli
        header_str = f"{'Model':<{model_col_w}} | {'Wariant':<{kind_col_w}} | "
        header_str += " | ".join(f"{h:^{rest_col_w}}" for h in metrics_headers + time_headers)

        total_w = len(header_str) + 4
        line = "═" * total_w

        print("\n" + line)
        print(f"║{'📊 ŚREDNIE WYNIKI ZBIORCZE'.center(total_w - 2)}║")
        print(line)
        print(f"| {header_str} |")
        print("-" * total_w)

        for (m_name, v_name), row in avg_df.iterrows():
            # 1. Nazwa modelu i wariantu (rozpakowane z krotki)
            row_str = f"{m_name:<{model_col_w}} | {v_name:<{kind_col_w}} | "

            # 2. Metryki (średnie)
            row_str += " | ".join(f"{v:^{rest_col_w}.8f}" for v in row)

            # 3. Czasy (pobierane z zsumowanego DF za pomocą klucza z krotki)
            t_s = times.loc[(m_name, v_name), 'Time [s]']
            t_m = times.loc[(m_name, v_name), 'Time [min]']
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
        columns = ["Trajektoria", "Model", "Dataset_kind"] + list(df.columns)

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
