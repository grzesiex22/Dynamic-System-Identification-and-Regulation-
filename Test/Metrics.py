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

    def save_all_to_file(self, dataset="Dataset1", folder="Reports"):
        # Używamy os.makedirs z exist_ok=True, aby stworzył wszystkie podfoldery
        full_path_dir = os.path.join(os.getcwd(), folder)
        os.makedirs(full_path_dir, exist_ok=True)

        path = os.path.join(full_path_dir, f"{dataset}_test_results.csv")

        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.to_csv(path)
        print(f"💾 Metryki zapisane w: {path}")

    def save_averages_to_file(self, dataset="Dataset1", folder="Reports"):
        """
        Zapisuje tylko uśrednione wyniki modeli do osobnego pliku.
        """
        avg_df = self.get_overall_average()
        if avg_df is None: return

        # Tworzenie folderu (ten sam schemat co wcześniej)
        target_dir = os.path.join(os.getcwd(), folder)
        os.makedirs(target_dir, exist_ok=True)

        path = os.path.join(target_dir, f"{dataset}_test_avg_results.csv")

        avg_df.to_csv(path)
        print(f"🏆 Średnie wyniki zapisane w: {path}")

    def show_averages(self):
        """
        Pobiera średnie i printuje je w sformatowanej tabeli.
        """
        avg_df = self.get_overall_average()
        if avg_df is None:
            print("Brak danych do uśrednienia.")
            return

        col_w = 12
        # Nagłówki (Model + nazwy metryk)
        headers = ["Model"] + list(avg_df.columns)
        header_str = " | ".join(f"{h:^{col_w}}" for h in headers)

        total_w = len(header_str) + 4
        line = "═" * total_w

        print("\n" + line)
        print(f"║{'📊 ŚREDNIE WYNIKI ZBIORCZE'.center(total_w - 3)}║")
        print(line)
        print(f"| {header_str} |")
        print("-" * total_w)

        for model_name, row in avg_df.iterrows():
            # Formatuje wartości do 8 miejsc po przecinku
            vals = [f"{model_name:<{col_w}}"] + [f"{v:^{col_w}.8f}" for v in row]
            print(f"| {' | '.join(vals)} |")

        print(line + "\n")

    def show_all(self):
        """
        Wyświetla porównanie w formie idealnie sformatowanej tabeli w konsoli.
        """
        if not self.results:
            print("Brak danych do wyświetlenia.")
            return None

        # 1. Tworzymy DataFrame do obliczeń i zwrotu
        df = pd.DataFrame.from_dict(self.results, orient='index')

        # 2. Ustawienia szerokości kolumn
        col_w = 13

        # Przygotowanie nagłówków (Trajektoria, Model + wszystkie metryki)
        columns = ["Trajektoria", "Model"] + list(df.columns)
        header_str = " | ".join(f"{col:^{col_w}}" for col in columns)

        # Dynamiczna szerokość tabeli na podstawie ilości metryk
        total_width = len(header_str) + 4
        thick_line = "═" * total_width
        thin_line = "=" * total_width

        # 3. Rysowanie nagłówka
        print("\n" + thick_line)
        print(f"║{'📊 SZCZEGÓŁOWE PORÓWNANIE METRYK'.center(total_width - 3)}║")
        print(thick_line)
        print(f"| {header_str} |")
        print(thin_line)

        # 4. Rysowanie wierszy z danymi
        last_traj = None
        for (traj, model), row in df.iterrows():
            # Gruba linia oddzielająca trajektorie
            if last_traj is not None and traj != last_traj:
                print(thin_line)

            # Formatyzowanie tekstu w komórkach
            traj_str = f"{traj}"
            row_vals = [f"{traj_str:<{col_w}}", f"{model:<{col_w}}"]
            row_vals += [f"{val:^{col_w}.8f}" for val in row]

            # Druk wiersza
            print(f"| {' | '.join(row_vals)} |")

            last_traj = traj

        print(thick_line + "\n")
        return df  # Zwracamy normalny czysty DataFrame