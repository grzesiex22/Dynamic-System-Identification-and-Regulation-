import numpy as np
from matplotlib import pyplot as plt

from SystemPlotter import SystemPlotter
from DataGenerator import DataGenerator
from SystemMLP import SystemMLP
from CoupledTanks import CoupledTanks1
from ImplementationOfMLP import OwnSystemMLP, Metrics


# -----------------------------
# 1. Definicja sygnału sterującego
# -----------------------------
def my_u(t):
    if t < 100: return 5.0
    if t < 300: return 7.0
    return 4.0


# -----------------------------
# 2. Funkcja pomocnicza do wypisywania metryk
# -----------------------------
def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")


# -----------------------------
# 3. Definicja obiektu dynamicznego
# -----------------------------
tanks = CoupledTanks1()
t_end = tanks.t_span[1]
dt = 0.5
print("System initialized")

# -----------------------------
# 4. Generacja danych treningowych
# -----------------------------
print("Generating training data...")
generator = DataGenerator(tanks, my_u, dt=dt)
train_data = generator.generate()  # zwraca SystemData

X_train, y_target_train = train_data.get_training_data()

print("Training data generated:")
print(f"train_data.y shape   = {train_data.y.shape}")
print(f"train_data.u shape   = {train_data.u.shape}")
print(f"train_data.t shape   = {train_data.t.shape}")
print(f"X_train shape        = {X_train.shape}")
print(f"y_target_train shape = {y_target_train.shape}")

# -----------------------------
# 5. Tworzymy i trenujemy MLP (gotowe)
# -----------------------------
# input_dim=3 (u(k), h1(k-1), h2(k-1)) -> output_dim=2 (h1(k), h2(k))
mlp = SystemMLP(input_dim=3, hidden_dim=64, output_dim=2)
mlp.train(train_data, lr=0.001, epochs=2000)


# -----------------------------
# 6. Symulacja testowa - przygotowanie danych
# -----------------------------
# Definiujemy nowy sygnał testowy (np. sinusoida)
# u_func_test = lambda t: 5.0 + 2.0 * np.sin(0.05 * t)
u_func_test = lambda t: my_u(t)  # to samo wymuszenie co w treningu
t_test = np.arange(0, t_end, dt)
u_test = np.array([u_func_test(ti) for ti in t_test]).reshape(-1, 1)

# Prawdziwe rozwiązanie (Ground Truth) dla testu
generator_test = DataGenerator(tanks, u_func_test, dt=dt)
true_data = generator_test.generate()

y_true = true_data.y  # Macierz [N x 2] (h1, h2)
X_all, dy_dt_true = train_data.get_training_data()
y0_test = y_true[0]   # Stan początkowy z prawdziwych danych

# -----------------------------
# 6. Symulacja testowa MLP (gotowe)
# -----------------------------
# Symulacja rekurencyjna MLP (Model używa własnych poprzednich wyjść)
y_sim_1 = mlp.simulate(u_new=u_test, y0=y0_test, dt=dt)

# Obliczamy pochodne (opcjonalnie do wykresów)
dy_dt_sim_1 = np.gradient(y_sim_1, dt, axis=0)

# -----------------------------
# 7. Tworzymy i trenujemy MLP (gotowe)
# -----------------------------
mlp2 = OwnSystemMLP(
    input_dim=3,
    hidden_layers=[64, 64],
    output_dim=2
)

mlp2.summary()

print("\nTraining own MLP...")
mlp2.train(
    X_train,
    y_target_train,
    epochs=2000,
    lr=0.01,
    val_ratio=0.2,
    shuffle=True,
    print_every=100
)

# -----------------------------
# 7. Symulacja testowa MLP (własne)
# -----------------------------
dy_dt_pred = mlp2.predict(X_train)
derivative_metrics = Metrics.evaluate(y_target_train, dy_dt_pred)
print_metrics("Metryki predykcji pochodnych dy/dt", derivative_metrics)

# Symulacja rekurencyjna MLP
print("\nRunning recursive simulation...")
y_sim_2 = mlp2.simulate(u_new=u_test, y0=y0_test, dt=dt)
print(f"y_sim shape = {y_sim_2.shape}")

# Metryki symulacji
simulation_metrics = Metrics.evaluate(y_true, y_sim_2)
print_metrics("Metryki symulacji wielokrokowej", simulation_metrics)

# Predykcja pochodnych do porównania
dy_dt_sim_2 = mlp2.predict(X_all)

# -----------------------------
# 8. Wykres porównawczy
# -----------------------------
# Tworzymy wersje "skrócone" do wykresu pochodnych
t_der = t_test[:-1]           # skracamy czas o 1 punkt
y_true_der = y_true[:-1]      # skracamy stany o 1 punkt (żeby pasowały do t_der)
u_der = u_test[:-1]           # skracamy sterowanie

SystemPlotter.plot(
    t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
    u=u_der,                  # Skrócone u
    y_true=y_true_der,        # Skrócone y_true
    dy_dt_true=dy_dt_true,    # To ma już 999 pkt
    y_sim_list=[y_sim_1[:-1]],  # Skracamy symulacje
    dy_dt_sim_list=[dy_dt_sim_1[:-1]],  # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
    legend_sim=["MLP PyTorch"],
    title="Porównanie trajektorii systemu (MLP PyTorch vs true)"
)

SystemPlotter.plot(
    t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
    u=u_der,                  # Skrócone u
    y_true=y_true_der,        # Skrócone y_true
    dy_dt_true=dy_dt_true,    # To ma już 999 pkt
    y_sim_list=[y_sim_2[:-1]],  # Skracamy symulacje
    dy_dt_sim_list=[dy_dt_sim_2],  # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
    legend_sim=["MLP Własne"],
    title="Porównanie trajektorii systemu (MLP Własne vs true)"
)

SystemPlotter.plot(
    t=t_der,                  # Przekazujemy skrócony czas (999 pkt)
    u=u_der,                  # Skrócone u
    y_true=y_true_der,        # Skrócone y_true
    dy_dt_true=dy_dt_true,    # To ma już 999 pkt
    y_sim_list=[y_sim_1[:-1], y_sim_2[:-1]], # Skracamy symulacje
    dy_dt_sim_list=[dy_dt_sim_1[:-1], dy_dt_sim_2], # dy_dt_sim_1 (z np.gradient) ma 1000, dy_dt_sim_2 ma 999
    legend_sim=["MLP PyTorch", "MLP Własne"],
    title="Porównanie trajektorii systemu (MLP PyTorch vs MLP własne vs true)"
)

plt.show(block=True)
