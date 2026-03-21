import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class SystemMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=None, output_dim=2):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(
    model,
    X,
    y,
    epochs=1000,
    lr=0.001,
    val_ratio=0.2,
    print_every=100
):
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_ratio))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        pred = model(X_train)
        train_loss = criterion(pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoka {epoch:4d} | "
                f"train: {train_loss.item():.6f} | "
                f"val: {val_loss.item():.6f}"
            )

    model.load_state_dict(best_state)
    print(f"\nNajlepszy val loss: {best_val_loss:.6f}")
    return model


def predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    return y_pred


def simulate(model, u_new, y0, dt):
    u_new = np.asarray(u_new, dtype=np.float32)

    if u_new.ndim == 1:
        u_new = u_new.reshape(-1, 1)

    y0 = np.asarray(y0, dtype=np.float32).flatten()

    n_points = len(u_new)
    output_dim = len(y0)

    y_sim = np.zeros((n_points, output_dim), dtype=np.float32)
    y_sim[0] = y0

    model.eval()
    with torch.no_grad():
        for i in range(1, n_points):
            u_curr = u_new[i - 1]
            y_prev = y_sim[i - 1]

            x_input = np.concatenate([u_curr, y_prev]).astype(np.float32)
            x_tensor = torch.tensor(x_input.reshape(1, -1), dtype=torch.float32)

            dy_dt_pred = model(x_tensor).numpy().flatten()
            y_sim[i] = y_prev + dy_dt_pred * dt
            y_sim[i] = np.maximum(y_sim[i], 0.0)

    return y_sim


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)