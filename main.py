import numpy as np
from matplotlib import pyplot as plt

from Dataset.DatasetHandler import DatasetHandler
from SystemPlotter import SystemPlotter
from Dataset.DatasetManager import DatasetManager
from SystemMLP import SystemMLP
from Objects.CoupledTanks import CoupledTanks1
from ImplementationOfMLP import OwnSystemMLP
from Metrics import Metrics


# -----------------------------
# 0. Parametry
# -----------------------------
t_end = 1000
dt = 0.5
amp_range = [3, 15]

# --- generator
generate = False
train_dataset_count = 100
val_dataset_count = 20
test_dataset_count = 20
dataset_folder = "Dataset2"

# -----------------------------
# 1. Definicja obiektu dynamicznego
# -----------------------------
tanks = CoupledTanks1(t_end=t_end)
print("System initialized")

# -----------------------------
# 2. Generator przebiegów
# -----------------------------
if generate:
    dataset_filename = f"train_dataset_{train_dataset_count}.npz"
    manager = DatasetManager(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    manager.create_dataset(n_trajectories=train_dataset_count, folder=dataset_folder, filename=dataset_filename)

    dataset_filename = f"val_dataset_{val_dataset_count}.npz"
    manager = DatasetManager(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    manager.create_dataset(n_trajectories=val_dataset_count, folder=dataset_folder, filename=dataset_filename)

    dataset_filename = f"test_dataset_{test_dataset_count}.npz"
    manager = DatasetManager(tanks, t_end=t_end, dt=dt, amp_range=amp_range)
    manager.create_dataset(n_trajectories=test_dataset_count, folder=dataset_folder, filename=dataset_filename)
    manager.show_random_signals(3)  # Podgląd

# -----------------------------
# 3A. Odczyt danych treningowych
# -----------------------------
dataset_filename = f"train_dataset_{train_dataset_count}.npz"
print(f"Loading training data from {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
X_train, y_target_train, t = DatasetHandler.load(folder=dataset_folder, filename=dataset_filename)

print("Training data loaded:")
print(f"X_train shape        = {X_train.shape}")         # SET x N-2 x 5
print(f"y_target_train shape = {y_target_train.shape}")  # SET x N-2 x 2

# -----------------------------
# 3B. Odczyt danych treningowych
# -----------------------------
dataset_filename = f"val_dataset_{val_dataset_count}.npz"
print(f"Loading validating data from {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
X_val, y_target_val, _ = DatasetHandler.load(folder=dataset_folder, filename=dataset_filename)

print("Validating data loaded:")
print(f"X_val shape        = {X_val.shape}")         # SET x N-2 x 5
print(f"y_target_val shape = {y_target_val.shape}")  # SET x N-2 x 2

# -----------------------------
# 3C. Odczyt danych treningowych
# -----------------------------
dataset_filename = f"test_dataset_{test_dataset_count}.npz"
print(f"Loading testing data from {dataset_filename}...")
# Wczytujemy wielką macierz (wszystkie trajektorie połączone)
X_test, y_target_test, _ = DatasetHandler.load(folder=dataset_folder, filename=dataset_filename)

print("Testing data loaded:")
print(f"X_test shape        = {X_test.shape}")         # SET x N-2 x 5
print(f"y_target_test shape = {y_target_test.shape}")  # SET x N-2 x 2

# -----------------------------
# 4. Tworzymy i trenujemy MLP (gotowe)
# -----------------------------
# Inicjalizacja modelu
mlp = SystemMLP(input_dim=5, hidden_dim=128, output_dim=2)

# Przekazujemy surowe macierze 3D (SET x N-2 x 5)
mlp.train(
    X_train,
    y_target_train,
    X_val,
    y_target_val,
    epochs=20,
    lr=0.0005  # Przy uczeniu trajektoria po trajektorii mniejszy LR jest bezpieczniejszy
)

# -----------------------------
# 5. Symulacja testowa na jednej trajektorii (MLP GOTOWE)
# -----------------------------
idx = 0  # Numer trajektorii ze zbioru testowego (0-199)

# --- Przygotowanie danych testowych (idx to numer trajektorii) ---
t_plot = t[0, 1:]                # Czas
u_plot = X_test[idx, :, 0]          # Sterowanie u
h_true = X_test[idx, :, 1:3]        # Prawdziwe poziomy h
dh_dt_true = y_target_test[idx]     # Prawdziwe pochodne dh/dt

# --- Symulacja modelu ---
print(f"\n🚀 Uruchamiam symulację dla trajektorii testowej nr {idx}...")
h0_test = h_true[0]
h_sim, dh_dt_sim = mlp.simulate(u_new=u_plot.reshape(-1, 1), h0=h0_test, dt=dt)

# -----------------------------
# 6. Metryki i Wykresy
# -----------------------------
# Porównujemy całą macierz [1998 x 2]
sim_metrics = Metrics.evaluate(dh_dt_true, dh_dt_sim)

print("\n=== METRYKI SYMULACJI (REKURENCYJNEJ) ===")
for key, value in sim_metrics.items():
    print(f"{key}: {value:.6f}")


# -----------------------------
# 9. Wykres
# -----------------------------
SystemPlotter.plot(
    t=t_plot,
    u=u_plot,
    y_true=h_true,
    dy_dt_true=dh_dt_true,
    y_sim_list=[h_sim],
    dy_dt_sim_list=[dh_dt_sim],
    legend_sim=["Model MLP (5 wejść)"],
    title="Weryfikacja modelu na danych testowych"
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
