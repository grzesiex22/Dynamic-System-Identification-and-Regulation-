import numpy as np


class SystemData:
    """
    Przechowuje dane systemu dynamicznego:
        - y(t)
        - u(t)
        - dy/dt
        - t
    Nie generuje danych ani nie trenuje modeli.
    """
    def __init__(self, y, u, t):
        self.y = np.array(y)
        self.u = np.array(u)
        self.t = np.array(t)
        self.dt = t[1]-t[0]
        self.dy_dt = np.gradient(self.y, self.dt, axis=0)

    def get_training_data(self):
        X = np.hstack([self.y, self.u])
        y_target = self.dy_dt
        return X, y_target

    def get_data_to_plot(self):
        return self.t, self.y, self.dy_dt, self.u