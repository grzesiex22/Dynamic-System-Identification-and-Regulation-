import numpy as np
from matplotlib import pyplot as plt

from Utills.SystemPlotter import SystemPlotter
from Dataset.DatasetCreator import DatasetCreator
from Dataset.DatasetReader import DatasetReader

from ML.SystemMLP import SystemMLP
from Objects.CoupledTanks import CoupledTanks1
from Utills.Metrics import Metrics


# -----------------------------
# 0. Parametry
# -----------------------------
t_end = 1000
dt = 0.5
amp_range = [3, 15]

# --- generator
generate = True
train_dataset_count = 100
val_dataset_count = 20
test_dataset_count = 20
dataset_folder = "Dataset3"

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
    dataset_creator.create_dataset(n_trajectories=train_dataset_count, folder=dataset_folder, filename=dataset_filename)

    dataset_filename = f"val_dataset_{val_dataset_count}.npz"
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    dataset_creator.create_dataset(n_trajectories=val_dataset_count, folder=dataset_folder, filename=dataset_filename)

    dataset_filename = f"test_dataset_{test_dataset_count}.npz"
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    dataset_creator.create_dataset(n_trajectories=test_dataset_count, folder=dataset_folder, filename=dataset_filename)
    dataset_creator.show_random_signals(3)  # Podgląd

# -----------------------------
# 3. Odczyt danych (Unified Reader)
# -----------------------------
print("\n"), print("-" * 100)
print(f"\n--- Ładowanie danych z folderu {dataset_folder} ---")

reader = DatasetReader()

# --- 3A. Trening ---
dataset_filename = f"train_dataset_{train_dataset_count}.npz"
print(f"\nŁadowanie danych treningowych z: {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
train_objects = reader.read_all(folder=dataset_folder, filename=dataset_filename)
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
val_objects = reader.read_all(folder=dataset_folder, filename=dataset_filename)

X_val = np.stack([obj.get_training_data()[0] for obj in val_objects])
Y_val = np.stack([obj.get_training_data()[1] for obj in val_objects])

print(f"✅ Dane walidacyjne załadowane z: {dataset_filename}")
print(f"X_val shape: {X_val.shape}, y_val shape: {Y_val.shape}")

# --- 3C. Testy ---
dataset_filename = f"test_dataset_{test_dataset_count}.npz"
print(f"\nŁadowanie danych testowych z: {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
test_objects = reader.read_all(folder=dataset_folder, filename=dataset_filename)

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
mlp = SystemMLP(input_dim=5, hidden_dim=128, output_dim=2)

# Przekazujemy surowe macierze 3D (SET x N-2 x 5)
mlp.train(
    X_train,
    Y_train,
    X_val,
    Y_val,
    epochs=50,
    lr=0.0005  # Przy uczeniu trajektoria po trajektorii mniejszy LR jest bezpieczniejszy
)

# -----------------------------
# 5. Symulacja testowa na trzech trajektoriach (MLP GOTOWE)
# -----------------------------
print("\n"), print("-" * 100)

for i in [0, 1, 3]:
    idx = i  # idx to numer trajektorii
    test_obj = test_objects[idx]

    # --- Przygotowanie danych do symulacji i do porównania (plot) ---
    t_plot, u_plot, h_true, dh_dt_true = test_obj.get_data_to_plot()
    t_to_sim, u_to_sim, h0_to_sim, dh_dt0_to_sim = test_obj.get_data_to_simulate()

    # --- Symulacja modelu ---
    print(f"\n🚀 Uruchamiam symulację dla trajektorii testowej nr {idx}...")
    sim_obj = mlp.simulate(t=t_to_sim, u_new=u_to_sim, h0=h0_to_sim, dh_dt0=dh_dt0_to_sim)
    t_sim, u_sim, h_sim, dh_dt_sim = sim_obj.get_data_to_plot()

    # -----------------------------
    # 5a. Metryki i Wykresy
    # -----------------------------
    # Porównujemy całą macierz [1998 x 2]
    sim_metrics = Metrics.evaluate(dh_dt_true, dh_dt_sim)

    print("\n=== METRYKI SYMULACJI (REKURENCYJNEJ) ===")
    for key, value in sim_metrics.items():
        print(f"{key}: {value:.6f}")

    # -----------------------------
    # 5b. Wykres
    # -----------------------------
    SystemPlotter.plot(
        t=t_plot,
        u=u_plot,
        y_true=h_true,
        dy_dt_true=dh_dt_true,
        y_sim_list=[h_sim],
        dy_dt_sim_list=[dh_dt_sim],
        legend_sim=["Model MLP (5 wejść)"],
        title=f"Weryfikacja modelu na danych testowych (trajektoria: {idx})"
    )

plt.show(block=True)

# # -----------------------------
# # 7. Symulacja testowa MLP (gotowe)
# # -----------------------------
# # Symulacja rekurencyjna MLP (Model używa własnych poprzednich wyjść)
# y_sim_1 = mlp.simulate(u_new=u_test, y0=y0_test, dt=dt)
#
# # Obliczamy pochodne (opcjonalnie do wykresów)
# dy_dt_sim_1 = np.gradient(y_sim_1, dt, axis=0)
#
# # -----------------------------
# # 8. Tworzymy i trenujemy MLP (gotowe)
# # -----------------------------
# mlp2 = OwnSystemMLP(
#     input_dim=3,
#     hidden_layers=[64, 64],
#     output_dim=2
# )
#
# mlp2.summary()
#
# print("\nTraining own MLP...")
# mlp2.train(
#     X_train,
#     y_target_train,
#     epochs=2000,
#     lr=0.01,
#     val_ratio=0.2,
#     shuffle=True,
#     print_every=100
# )
#
# # -----------------------------
# # 9. Symulacja testowa MLP (własne)
# # -----------------------------
# dy_dt_sim_2 = mlp2.predict(X_train)
# derivative_metrics = Metrics.evaluate(y_target_train, dy_dt_sim_2)
# print_metrics("Metryki predykcji pochodnych dy/dt", derivative_metrics)
#
# # Symulacja rekurencyjna MLP
# print("\nRunning recursive simulation...")
# y_sim_2 = mlp2.simulate(u_new=u_test, y0=y0_test, dt=dt)
# print(f"y_sim shape = {y_sim_2.shape}")
#
# # Metryki symulacji
# simulation_metrics = Metrics.evaluate(y_true, y_sim_2)
# print_metrics("Metryki symulacji wielokrokowej y(t)", simulation_metrics)
#

# # -----------------------------
# # 10. Wykresy porównawcze
# # -----------------------------
# # Tworzymy wersje "skrócone" do wykresu pochodnych
# t_der = t_test[:-1]           # skracamy czas o 1 punkt
# y_true_der = y_true[:-1]      # skracamy stany o 1 punkt (żeby pasowały do t_der)
# u_der = u_test[:-1]           # skracamy sterowanie
#
# SystemPlotter.plot(
#     t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
#     u=u_der,                  # Skrócone u
#     y_true=y_true_der,        # Skrócone y_true
#     dy_dt_true=dy_dt_true,    # To ma już 999 pkt
#     y_sim_list=[y_sim_1[:-1]],  # Skracamy symulacje
#     dy_dt_sim_list=[dy_dt_sim_1[:-1]],  # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
#     legend_sim=["MLP PyTorch"],
#     title="Porównanie trajektorii systemu (MLP PyTorch vs true)"
# )
#
# SystemPlotter.plot(
#     t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
#     u=u_der,                  # Skrócone u
#     y_true=y_true_der,        # Skrócone y_true
#     dy_dt_true=dy_dt_true,    # To ma już 999 pkt
#     y_sim_list=[y_sim_2[:-1]],  # Skracamy symulacje
#     dy_dt_sim_list=[dy_dt_sim_2],  # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
#     legend_sim=["MLP Własne"],
#     title="Porównanie trajektorii systemu (MLP Własne vs true)"
# )
#
# SystemPlotter.plot(
#     t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
#     u=u_der,                  # Skrócone u
#     y_true=y_true_der,        # Skrócone y_true
#     dy_dt_true=dy_dt_true,    # To ma już 999 pkt
#     y_sim_list=[y_sim_1[:-1], y_sim_2[:-1]], # Skracamy symulacje
#     dy_dt_sim_list=[dy_dt_sim_1[:-1], dy_dt_sim_2], # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
#     legend_sim=["MLP PyTorch", "MLP Własne"],
#     title="Porównanie trajektorii systemu (MLP PyTorch vs MLP własne vs true)"
# )
#
# plt.show(block=True)
