import numpy as np
from matplotlib import pyplot as plt

from Utills.SystemPlotter import SystemPlotter
from Dataset.DatasetCreator import DatasetCreator
from Dataset.DatasetReader import DatasetReader

from ML.SystemMLP import SystemMLP
from Objects.CoupledTanks import CoupledTanks1
from Utills.Metrics import Metrics, MetricsSummarizer
from ML.ImplementationOfMLP import OwnSystemMLP


# -----------------------------
# 0. Parametry
# -----------------------------
t_end = 1000
dt = 0.5
amp_range = [3, 15]

# --- generator
generate = False
train_dataset_count = 1000
val_dataset_count = 200
test_dataset_count = 200
dataset_name = "Dataset2"

# --- model
train_and_save = True
load = False
epochs = 100

# -----------------------------
# 1. Definicja obiektu dynamicznego
# -----------------------------
tanks = CoupledTanks1(t_end=t_end)
print("\n--- System zainicjalizowany ---")

# -----------------------------
# 2. Generator przebiegów
# -----------------------------
print("\n"), print("-" * 100)

if generate:
    dataset_filename = f"train_dataset_{train_dataset_count}.npz"
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    dataset_creator.create_dataset(n_trajectories=train_dataset_count, folder=dataset_name, filename=dataset_filename)

    dataset_filename = f"val_dataset_{val_dataset_count}.npz"
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    dataset_creator.create_dataset(n_trajectories=val_dataset_count, folder=dataset_name, filename=dataset_filename)

    dataset_filename = f"test_dataset_{test_dataset_count}.npz"
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    dataset_creator.create_dataset(n_trajectories=test_dataset_count, folder=dataset_name, filename=dataset_filename)
    dataset_creator.show_random_signals(3)  # Podgląd

# -----------------------------
# 3. Odczyt danych (Unified Reader)
# -----------------------------
print("\n"), print("-" * 100)
print(f"\n--- Ładowanie danych z folderu {dataset_name} ---")

reader = DatasetReader()

# --- 3A. Trening ---
dataset_filename = f"train_dataset_{train_dataset_count}.npz"
print(f"\nŁadowanie danych treningowych z: {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
train_objects = reader.read_all(folder=dataset_name, filename=dataset_filename)
reader.show_random_signals()

# Przygotowanie macierzy do uczenia (SET x N-2 x 5)
X_train = np.stack([obj.get_training_data()[0] for obj in train_objects])
Y_train = np.stack([obj.get_training_data()[1] for obj in train_objects])

print(f"✅ Dane treninigowe załadowane z: {dataset_filename}")
print(f"X_train shape: {X_train.shape}, y_train shape: {Y_train.shape}")

# --- 3B. Walidacja ---
dataset_filename = f"val_dataset_{val_dataset_count}.npz"
print(f"\nŁadowanie danych walidacyjnych z: {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
val_objects = reader.read_all(folder=dataset_name, filename=dataset_filename)

X_val = np.stack([obj.get_training_data()[0] for obj in val_objects])
Y_val = np.stack([obj.get_training_data()[1] for obj in val_objects])

print(f"✅ Dane walidacyjne załadowane z: {dataset_filename}")
print(f"X_val shape: {X_val.shape}, y_val shape: {Y_val.shape}")

# --- 3C. Testy ---
dataset_filename = f"test_dataset_{test_dataset_count}.npz"
print(f"\nŁadowanie danych testowych z: {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
test_objects = reader.read_all(folder=dataset_name, filename=dataset_filename)

X_test = np.stack([obj.get_training_data()[0] for obj in test_objects])
Y_test = np.stack([obj.get_training_data()[1] for obj in test_objects])

print(f"✅ Dane testowe załadowane z: {dataset_filename}")
print(f"X_test shape: {X_test.shape}, y_test shape: {Y_test.shape}")

# --- 3D. CZAS ---
t = train_objects[0].t

# -----------------------------
# 4. Tworzymy i trenujemy MLP (gotowe)
# -----------------------------
print("\n"), print("-" * 100)

# Inicjalizacja modelu
troch_mlp = SystemMLP(input_dim=5, hidden_dim=128, output_dim=2)

if load:
    troch_mlp.load_model(dataset=dataset_name)

if train_and_save:
    troch_mlp.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs=epochs,
        lr=0.0005  # Przy uczeniu trajektoria po trajektorii mniejszy LR jest bezpieczniejszy
    )
    troch_mlp.save_model(dataset=dataset_name)

# -----------------------------
# 5. Tworzymy i trenujemy MLP (własne)
# -----------------------------
print("\n"), print("-" * 100)

own_mlp = OwnSystemMLP(input_dim=5, hidden_dim=128, output_dim=2)

if load:
    own_mlp.load_model(dataset=dataset_name)

if train_and_save:
    own_mlp.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs=epochs,
        lr=0.0005,
    )
    own_mlp.save_model(dataset=dataset_name)

# -----------------------------
# 6. Symulacja testowa na trzech trajektoriach (MLP GOTOWE - MLP WŁASNE)
# -----------------------------
print("\n"), print("-" * 100)

# Inicjalizacja metryk
summarizer = MetricsSummarizer()

for i in [0, 1, 3]:
    idx = i  # idx to numer trajektorii
    test_obj = test_objects[idx]

    # --- Przygotowanie danych do symulacji i do porównania (plot) ---
    t_plot, u_plot, h_true, dh_dt_true = test_obj.get_data_to_plot()
    t_to_sim, u_to_sim, h0_to_sim, dh_dt0_to_sim = test_obj.get_data_to_simulate()

    # --- Symulacja modelu - TORCH MLP ---
    print(f"\n🚀 Uruchamiam symulację WŁASNEGO MLP dla trajektorii testowej nr {idx}...")
    sim_torch_mlp_obj = troch_mlp.simulate(t=t_to_sim, u_new=u_to_sim, h0=h0_to_sim, dh_dt0=dh_dt0_to_sim)
    t_sim_torch_mlp, u_sim_torch_mlp, h_sim_torch_mlp, dh_dt_sim_torch_mlp = sim_torch_mlp_obj.get_data_to_plot()

    # --- Symulacja modelu - OWN MLP ---
    print(f"🚀 Uruchamiam symulację WŁASNEGO MLP dla trajektorii testowej nr {idx}...")
    sim_own_mlp_obj = own_mlp.simulate(t=t_to_sim, u_new=u_to_sim, h0=h0_to_sim, dh_dt0=dh_dt0_to_sim)
    t_sim_own_mlp, u_sim_own_mlp, h_sim_own_mlp, dh_dt_sim_own_mlp = sim_own_mlp_obj.get_data_to_plot()

    # --- Obliczanie metryk ---
    torch_mlp_metrics = Metrics.evaluate(dh_dt_true, dh_dt_sim_torch_mlp)
    own_mlp_metrics = Metrics.evaluate(dh_dt_true, dh_dt_sim_own_mlp)

    # --- Dodawanie do podsumowania ---
    summarizer.add_metrics(i, "Torch_MLP", torch_mlp_metrics)
    summarizer.add_metrics(i, "Own_MLP", own_mlp_metrics)

    # --- Wykres ---
    SystemPlotter.plot(
        t=t_plot,
        u=u_plot,
        y_true=h_true,
        dy_dt_true=dh_dt_true,
        y_sim_list=[h_sim_own_mlp, h_sim_torch_mlp],
        dy_dt_sim_list=[dh_dt_sim_own_mlp, dh_dt_sim_torch_mlp],
        legend_sim=["OWN MLP", "TORCH MLP"],
        title=f"Weryfikacja modelu na danych testowych (trajektoria: {idx})"
    )

#  Wyświetlenie tabeli metryk na koniec
comparison_df = summarizer.summarize()

# Opcjonalnie: Znajdź najlepszy model dla Trajektorii 0
best_name, best_val = summarizer.get_best_model("MSE")
print(f"🏆 Najlepszy wynik ogólny (MSE): {best_name} z wartością {best_val:.8f}")


plt.show(block=True)

