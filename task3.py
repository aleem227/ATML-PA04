# task3_dirichlet_fedavg.py
# Task 3: FedAvg under label heterogeneity (Dirichlet alpha)
# - Dataset: CIFAR-10
# - Clients: M (default 5)
# - Non-IID split via Dirichlet(alpha) over labels
# - Metrics per round: global test accuracy, average client drift, regret
# - Plots: Accuracy vs rounds (per alpha), Drift vs rounds (per alpha), Regret vs rounds (per alpha)

import argparse
import copy
import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# Reproducibility & Device
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model
# -----------------------------
class SmallCIFAR10CNN(nn.Module):
    """Small CNN suitable for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Helper functions
# -----------------------------
def vectorize_params(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for p in model.parameters()])


def evaluate(model: nn.Module, loader: DataLoader, device, criterion=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    acc = correct / total
    loss = (loss_sum / total) if criterion is not None else None
    return loss, acc


def average_state_dicts(state_dicts, weights=None):
    """Weighted average of state_dicts. weights must sum to 1."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    out = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() * w for sd, w in zip(state_dicts, weights)], dim=0)
        out[k] = stacked.sum(dim=0)
    return out


# -----------------------------
# Dirichlet Non-IID Split
# -----------------------------
def dirichlet_split_indices(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 123):
    """Split dataset indices among clients using a Dirichlet(alpha) distribution."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = idx_by_class[c]
        n_c = len(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (proportions * n_c).astype(int)
        remainder = n_c - splits.sum()
        if remainder > 0:
            frac = proportions * n_c - splits
            order = np.argsort(-frac)
            for k in order[:remainder]:
                splits[k] += 1
        start = 0
        for i in range(num_clients):
            take = int(splits[i])
            if take > 0:
                client_indices[i].extend(idx_c[start:start + take])
                start += take
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices


def get_cifar10_dirichlet_loaders(num_clients, alpha, batch_size, data_root="./data", seed=123):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    labels = np.array(trainset.targets)
    client_idx_lists = dirichlet_split_indices(labels, num_clients, alpha, seed)
    client_subsets = [Subset(trainset, idxs) for idxs in client_idx_lists]
    client_loaders = [DataLoader(s, batch_size=batch_size, shuffle=True) for s in client_subsets]
    client_sizes = [len(s) for s in client_subsets]
    test_loader = DataLoader(testset, batch_size=512, shuffle=False)
    return client_loaders, client_sizes, test_loader


# -----------------------------
# Local client training
# -----------------------------
def local_train(model, loader, device, epochs, lr, grad_clip=None):
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    return model.state_dict()


# -----------------------------
# One FedAvg round
# -----------------------------
def fedavg_round(global_model, client_loaders, client_sizes, device, K, lr, grad_clip=None):
    num_clients = len(client_loaders)
    theta_g_vec = vectorize_params(global_model).to(device)
    local_states, weights, drifts = [], [], []

    for idx in range(num_clients):
        client_model = copy.deepcopy(global_model).to(device)
        local_sd = local_train(client_model, client_loaders[idx], device, K, lr, grad_clip)
        local_states.append(local_sd)
        weights.append(client_sizes[idx])
        client_model.load_state_dict(local_sd, strict=True)
        theta_i_vec = vectorize_params(client_model).to(device)
        drift = torch.linalg.vector_norm(theta_i_vec - theta_g_vec).item()
        drifts.append(drift)

    total = float(sum(weights))
    w_norm = [w / total for w in weights]
    new_global_sd = average_state_dicts(local_states, w_norm)
    avg_drift = float(sum(drifts) / len(drifts))
    return new_global_sd, avg_drift


# -----------------------------
# Run experiment for one alpha
# -----------------------------
def run_experiment(alpha, base_state, num_clients, K, rounds, lr, batch_size, device):
    client_loaders, client_sizes, test_loader = get_cifar10_dirichlet_loaders(
        num_clients=num_clients, alpha=alpha, batch_size=batch_size
    )
    global_model = SmallCIFAR10CNN().to(device)
    global_model.load_state_dict(base_state, strict=True)
    criterion = nn.CrossEntropyLoss()

    loss0, acc0 = evaluate(global_model, test_loader, device, criterion)
    print(f"[alpha={alpha}] Round 0 | Acc={acc0:.4f}")

    min_loss = float("inf")
    history = []
    for t in range(1, rounds + 1):
        new_state, avg_drift = fedavg_round(global_model, client_loaders, client_sizes, device, K, lr)
        global_model.load_state_dict(new_state, strict=True)
        loss, acc = evaluate(global_model, test_loader, device, criterion)
        min_loss = min(min_loss, loss)
        regret = loss - min_loss
        print(f"[alpha={alpha}] Round {t:02d} | Acc={acc:.4f} | Drift={avg_drift:.4f} | Regret={regret:.6f}")
        history.append({"round": t, "acc": acc, "drift": avg_drift, "regret": regret})
    return history


# -----------------------------
# Plotting
# -----------------------------
def try_plot(results, title, ykey, ylabel=None):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        for alpha, hist in results.items():
            x = [d["round"] for d in hist]
            y = [d[ykey] for d in hist]
            plt.plot(x, y, label=f"alpha={alpha}")
        plt.xlabel("Round")
        plt.ylabel(ylabel or ykey)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Plot skipped: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 3: FedAvg under Dirichlet non-IID (vary alpha).")
    parser.add_argument("--alphas", type=str, default="0.05,0.2,1,100")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"[Task 3] Device: {device}")

    base_model = SmallCIFAR10CNN().to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    results = {}

    for alpha in alphas:
        hist = run_experiment(alpha, base_state, args.num_clients, args.K, args.rounds, args.lr, args.batch_size, device)
        results[alpha] = hist

    if not args.no_plots:
        try_plot(results, "Accuracy vs Rounds (vary alpha)", ykey="acc", ylabel="Accuracy")
        try_plot(results, "Average Drift vs Rounds (vary alpha)", ykey="drift", ylabel="Average Drift")
        try_plot(results, "Regret vs Rounds (vary alpha)", ykey="regret", ylabel="Regret")

    print("\n=== Summary ===")
    for alpha in alphas:
        last = results[alpha][-1]
        print(f"[alpha={alpha}] Acc={last['acc']:.4f} | Drift={last['drift']:.4f} | Regret={last['regret']:.6f}")


if __name__ == "__main__":
    main()
