import numpy as np
from SystemPlotter import SystemPlotter
from DataGenerator import DataGenerator
from SystemData import SystemData
from SystemMLP import SystemMLP
from CoupledTanks import CoupledTanks1


# -----------------------------
# 1. Definicja sygnału sterującego
# -----------------------------
def my_u(t):
    if t < 100: return 5.0
    if t < 300: return 7.0
    return 4.0


# -----------------------------
# 2. Definicja obiektu dynamicznego
# -----------------------------
tanks = CoupledTanks1()

t_end = tanks.t_span[1]
dt = 0.5
print("System inited")

# -----------------------------
# 3. Generacja danych treningowych
# -----------------------------
print("Data generating...")
generator = DataGenerator(tanks, my_u, dt=dt)
train_data = generator.generate()  # zwraca SystemData


# -----------------------------
# 4. Tworzymy i trenujemy MLP
# -----------------------------
# input_dim=3 (u(k), h1(k-1), h2(k-1)) -> output_dim=2 (h1(k), h2(k))
# Uwaga: Dostosuj input_dim w SystemMLP zależnie od tego, jak zaimplementowałeś warstwę wejściową
mlp = SystemMLP(input_dim=3, hidden_dim=64, output_dim=2)
mlp.train(train_data, lr=0.001, epochs=2000)


# -----------------------------
# 4. Symulacja testowa (Inne wymuszenie niż w treningu)
# -----------------------------
# Definiujemy nowy sygnał testowy (np. sinusoida)
# u_func_test = lambda t: 5.0 + 2.0 * np.sin(0.05 * t)
u_func_test = lambda t: my_u(t)
t_test = np.arange(0, t_end, dt)
u_test = np.array([u_func_test(ti) for ti in t_test]).reshape(-1, 1)

# Prawdziwe rozwiązanie (Ground Truth) dla testu
generator_test = DataGenerator(tanks, u_func_test, dt=dt)
true_data = generator_test.generate()

y_true = true_data.y  # Macierz [N x 2] (h1, h2)
y0_test = y_true[0]   # Stan początkowy z prawdziwych danych

# Symulacja rekurencyjna MLP (Model używa własnych poprzednich wyjść)
# To jest właśnie "predykcja w długim horyzoncie"
y_sim = mlp.simulate(u_new=u_test, y0=y0_test, dt=dt)

# Obliczamy pochodne (opcjonalnie do wykresów)
dy_dt_true = np.gradient(y_true, dt, axis=0)
dy_dt_sim = np.gradient(y_sim, dt, axis=0)

# -----------------------------
# 5. Wykres porównawczy
# -----------------------------
SystemPlotter.plot(
    t=t_test,
    y_true=y_true,
    y_sim=y_sim,
    dy_dt_true=dy_dt_true,
    dy_dt_sim=dy_dt_sim,
    u=u_test,
    title="Porównanie trajektorii systemu (MLP vs true)"
)