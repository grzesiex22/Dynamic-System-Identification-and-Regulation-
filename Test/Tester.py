from tqdm import tqdm
from Test.Metrics import Metrics

class Tester:
    def __init__(self, test_objects):
        self.test_objects = test_objects

    def run(self, models, summarizer):
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
                _, _, _, dh_dt_true = test_obj.get_data_to_plot()

                # 2. Symulacja (rekurencyjna)
                sim_obj = m_obj.simulate(t=t_to_sim, u_new=u_to_sim, h0=h0_to_sim, dh_dt0=dh_dt0_to_sim)
                _, _, _, dh_dt_sim = sim_obj.get_data_to_plot()

                # 3. Obliczanie metryk
                sim_metrics = Metrics.evaluate(dh_dt_true, dh_dt_sim)

                # 4. Dodawanie do wspólnego summarizera
                summarizer.add_metrics(i, m_name, sim_metrics)