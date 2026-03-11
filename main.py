import numpy as np
from SystemPlotter import SystemPlotter
from DataGenerator import DataGenerator
from SystemData import SystemData
from SystemMLP import SystemMLP


# -----------------------------
# 1. Definicja równania dynamicznego
# -----------------------------
def f(y, u):
    """
    Przykładowe równanie 1D:
    dy/dt = -2*y + 3*u
    """
    return -2*y + 3*u

# -----------------------------
# 2. Generacja danych treningowych
# -----------------------------
y0 = [0]  # stan początkowy
t_end = 5
dt = 0.01
u_func_train = lambda t: np.sin(2*t)

generator = DataGenerator(f=f, y0=y0, u_func=u_func_train, t_end=t_end, dt=dt)
train_data = generator.generate()  # zwraca SystemData

# -----------------------------
# 3. Tworzymy i trenujemy MLP
# -----------------------------
mlp = SystemMLP(input_dim=2, hidden_dim=16)
mlp.train(train_data, lr=0.01, epochs=3000, print_every=500)

# -----------------------------
# 4. Symulacja dla nowego wymuszenia
# -----------------------------
u_func_test = lambda t: np.cos(3*t)
t_test = np.arange(0, t_end, dt)
u_test = np.array([u_func_test(ti) for ti in t_test]).reshape(-1,1)

y_sim = mlp.simulate(u_new=u_test, y0=y0[0], dt=dt)
dy_dt_sim = np.gradient(y_sim, dt)

# Obliczamy prawdziwe y(t) dla porównania
generator_test = DataGenerator(f=f, y0=y0, u_func=u_func_test, t_end=t_end, dt=dt)
true_data = generator_test.generate()
y_true = true_data.y
dy_dt_true = true_data.dy_dt

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