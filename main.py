import numpy as np
import matplotlib.pyplot as plt

from DataGenerator import DataGenerator
from CoupledTanks import CoupledTanks1
from ImplementationOfMLP import SystemMLP, Metrics


# -----------------------------
# 1. Definicja sygnału sterującego
# -----------------------------
def my_u(t):
    if t < 100:
        return 5.0
    if t < 300:
        return 7.0
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
train_data = generator.generate()

X_train, y_target_train = train_data.get_training_data()

print("Training data generated:")
print(f"train_data.y shape   = {train_data.y.shape}")
print(f"train_data.u shape   = {train_data.u.shape}")
print(f"train_data.t shape   = {train_data.t.shape}")
print(f"X_train shape        = {X_train.shape}")
print(f"y_target_train shape = {y_target_train.shape}")


# -----------------------------
# 5. Tworzenie i trening MLP
# -----------------------------
mlp = SystemMLP(
    input_dim=3,
    hidden_layers=[64, 64],
    output_dim=2
)

mlp.summary()

print("\nTraining MLP...")
mlp.train(
    X_train,
    y_target_train,
    epochs=2000,
    lr=0.001,
    val_ratio=0.2,
    shuffle=True,
    print_every=100
)


# -----------------------------
# 6. Ewaluacja predykcji pochodnych
# -----------------------------
dy_dt_pred = mlp.predict(X_train)
derivative_metrics = Metrics.evaluate(y_target_train, dy_dt_pred)
print_metrics("Metryki predykcji pochodnych dy/dt", derivative_metrics)


# -----------------------------
# 7. Symulacja testowa
# -----------------------------
def u_func_test(t):
    return my_u(t)


t_test = np.arange(0, t_end, dt)
u_test = np.array([u_func_test(ti) for ti in t_test], dtype=np.float64).reshape(-1, 1)

print("\nGenerating test data...")
generator_test = DataGenerator(tanks, u_func_test, dt=dt)
true_data = generator_test.generate()

y_true = true_data.y
y0_test = y_true[0]

print("Test data generated:")
print(f"y_true shape = {y_true.shape}")
print(f"u_test shape = {u_test.shape}")
print(f"y0_test      = {y0_test}")


# -----------------------------
# 8. Symulacja rekurencyjna MLP
# -----------------------------
print("\nRunning recursive simulation...")
y_sim = mlp.simulate(u_new=u_test, y0=y0_test, dt=dt)

print(f"y_sim shape = {y_sim.shape}")


# -----------------------------
# 9. Metryki symulacji
# -----------------------------
simulation_metrics = Metrics.evaluate(y_true, y_sim)
print_metrics("Metryki symulacji wielokrokowej", simulation_metrics)


# -----------------------------
# 10. Podgląd wyników
# -----------------------------
print("\nDone.")
print("\nPierwsze 5 próbek:")
print("y_true[:5] =")
print(y_true[:5])
print("\ny_sim[:5] =")
print(y_sim[:5])


# -----------------------------
# 11. Predykcja pochodnych do porównania
# -----------------------------
X_all, dy_dt_true = train_data.get_training_data()
dy_dt_pred = mlp.predict(X_all)

t_der = train_data.t[:-1]


# -----------------------------
# 12. Wykresy
# -----------------------------
plt.figure(figsize=(12, 12))

# 1. Sygnał wejściowy
plt.subplot(5, 1, 1)
plt.plot(t_test, u_test[:, 0], label="u(t)")
plt.title("Sygnał wejściowy")
plt.ylabel("u")
plt.grid(True)
plt.legend()

# 2. h1: prawdziwy vs symulowany
plt.subplot(5, 1, 2)
plt.plot(t_test, y_true[:, 0], label="h1 true")
plt.plot(t_test, y_sim[:, 0], "--", label="h1 MLP")
plt.title("Porównanie wyjścia h1")
plt.ylabel("h1")
plt.grid(True)
plt.legend()

# 3. h2: prawdziwy vs symulowany
plt.subplot(5, 1, 3)
plt.plot(t_test, y_true[:, 1], label="h2 true")
plt.plot(t_test, y_sim[:, 1], "--", label="h2 MLP")
plt.title("Porównanie wyjścia h2")
plt.ylabel("h2")
plt.grid(True)
plt.legend()

# 4. dh1/dt: prawdziwa pochodna vs przewidziana
plt.subplot(5, 1, 4)
plt.plot(t_der, dy_dt_true[:, 0], label="dh1/dt true")
plt.plot(t_der, dy_dt_pred[:, 0], "--", label="dh1/dt pred")
plt.title("Porównanie pochodnej dh1/dt")
plt.ylabel("dh1/dt")
plt.grid(True)
plt.legend()

# 5. dh2/dt: prawdziwa pochodna vs przewidziana
plt.subplot(5, 1, 5)
plt.plot(t_der, dy_dt_true[:, 1], label="dh2/dt true")
plt.plot(t_der, dy_dt_pred[:, 1], "--", label="dh2/dt pred")
plt.title("Porównanie pochodnej dh2/dt")
plt.xlabel("Czas [s]")
plt.ylabel("dh2/dt")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()