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

    def summarize(self):
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