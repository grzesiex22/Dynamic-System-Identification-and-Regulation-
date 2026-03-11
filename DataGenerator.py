from SystemData import SystemData
import numpy as np


class DataGenerator:
    """
    Generuje trajektorie systemu dynamicznego z funkcji różniczkowej f(y, u).
    """

    def __init__(self, f, y0, u_func, t_end=5, dt=0.01):
        self.f = f
        self.y0 = np.array(y0)
        self.u_func = u_func
        self.dt = dt
        self.t = np.arange(0, t_end, dt)
        self.n_points = len(self.t)

    def generate(self):
        n_dim = len(self.y0)
        y = np.zeros((self.n_points, n_dim))
        u = np.zeros((self.n_points, len(np.atleast_1d(self.u_func(0)))))
        y[0] = self.y0

        for i, ti in enumerate(self.t):
            u[i] = self.u_func(ti)
        for i in range(1, self.n_points):
            dy = self.f(y[i - 1], u[i - 1])
            y[i] = y[i - 1] + dy * self.dt

        return SystemData(y, u, self.t)