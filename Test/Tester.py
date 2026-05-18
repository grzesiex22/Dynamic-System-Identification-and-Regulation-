from tqdm import tqdm
import time
from Test.Metrics import Metrics


class Tester:
    def __init__(self, test_objects, clean_reference=None):
        """
        test_objects: obiekty wejściowe dla modelu (mogą mieć szum)
        clean_reference: idealne trajektorie (Ground Truth) do liczenia metryk
        """
        self.test_objects = test_objects
        # Jeśli nie podano referencji, używamy obiektów wejściowych (dla wariantu CLEAN)
        self.clean_reference = clean_reference if clean_reference is not None else test_objects

    def run(self, models, summarizer_dy, summarizer_y=None):
        """
        models: lista słowników [{"obj": model, "name": "ModelA"}, ...]
        summarizer: wspólny obiekt MetricsSummarizer
        """
        for model in models:
            m_obj = model["obj"]
            m_name = model["name"]

            # Pasek postępu po trajektoriach dla danego modelu
            test_bar = tqdm(self.test_objects, desc=f"Testowanie {m_name}", unit="traj")

            for i, test_obj in enumerate(test_bar):
                # 1. Pobieranie danych
                t_to_sim, u_to_sim, h0_to_sim, dh_dt0_to_sim = test_obj.get_data_to_simulate()

                # 2. Pobieranie danych referencyjnych (ZAWSZE CLEAN)
                clean_obj = self.clean_reference[i]
                t_true_clean, _, h_true_clean, dh_dt_true_clean = clean_obj.get_data_to_plot()

                # 3. Symulacja (rekurencyjna)
                start_time = time.perf_counter()
                sim_obj = m_obj.simulate(t=t_to_sim, u_new=u_to_sim, h0=h0_to_sim, dh_dt0=dh_dt0_to_sim)
                end_time = time.perf_counter()
                duration = end_time - start_time

                _, _, h_sim, dh_dt_sim = sim_obj.get_data_to_plot()

                # 4. Obliczanie metryk pochodnych
                sim_metrics_dy = Metrics.evaluate(dh_dt_true_clean, dh_dt_sim,
                                                  t=t_true_clean, suffixes=["_dh1_dt", "_dh2_dt"])
                sim_metrics_dy['Time [s]'] = duration
                sim_metrics_dy['Time [min]'] = duration/60

                # 5. Dodawanie do wspólnego summarizera
                summarizer_dy.add_metrics(i, m_name, sim_metrics_dy)

                if summarizer_y:
                    # 6. Obliczanie metryk wartości (nie pochodne)
                    sim_metrics_y = Metrics.evaluate(h_true_clean, h_sim,
                                                     t=t_true_clean, suffixes=["_h1", "_h2"])
                    sim_metrics_y['Time [s]'] = duration
                    sim_metrics_y['Time [min]'] = duration/60

                    # 7. Dodawanie do wspólnego summarizera
                    summarizer_y.add_metrics(i, m_name, sim_metrics_y)
