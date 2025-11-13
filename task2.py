# fedavg_task2.py
# Task 2: FedAvg with variable K (local epochs) and client sampling fractions.
# - Dataset: CIFAR-10 (easy to swap to MNIST if preferred)
# - Model: small CNN
# - Metrics tracked per round:
#     * test accuracy
#     * average client drift (mean ||theta_i^K - theta_global||_2)
#     * communication cost (# of client updates transmitted)
# - Experiments:
#     1) Vary K with full participation
#     2) Vary sampling fraction with fixed K

import argparse
import copy
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Weighted average of a list of state_dicts (same keys). Weights sum to 1."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    out = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() * w for sd, w in zip(state_dicts, weights)], dim=0)
        out[k] = stacked.sum(dim=0)
    return out

# -----------------------------
# Models
# -----------------------------
class SmallCIFAR10CNN(nn.Module):
    """A small CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
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

# (Optional) MNIST MLP if you want to switch datasets quickly
class SmallMNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Data
# -----------------------------
def get_cifar10_loaders_iid(num_clients, batch_size, data_root="./data", iid=True, equal_split=True):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    if equal_split:
        # Equal IID partitions
        per_client = len(trainset) // num_clients
        lengths = [per_client] * num_clients
        remainder = len(trainset) - per_client * num_clients
        for i in range(remainder):
            lengths[i] += 1
        splits = random_split(trainset, lengths, generator=torch.Generator().manual_seed(123))
        client_loaders = [DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=False) for s in splits]
        sizes = [len(s) for s in splits]
    else:
        # Weighted or unequal splits can be added here
        # For now, keep equal for clean weighting
        raise NotImplementedError

    test_loader = DataLoader(testset, batch_size=512, shuffle=False)
    return client_loaders, sizes, test_loader

# -----------------------------
# Local client training
# -----------------------------
def local_train(model, loader, device, epochs, lr, weight_decay=0.0, momentum=0.0):
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# -----------------------------
# FedAvg round
# -----------------------------
def fedavg_round(global_model, client_loaders, client_sizes, device, K_local_epochs, lr, sample_fraction=1.0,
                 momentum=0.0, weight_decay=0.0):
    """
    Perform one FedAvg communication round:
      1) Sample a subset of clients (fraction)
      2) Send global to each sampled client
      3) Each client trains K epochs locally
      4) Aggregate weighted by data size
    Returns:
      new_global_state_dict, avg_client_drift, num_participants
    """
    num_clients = len(client_loaders)
    m = max(1, int(math.ceil(sample_fraction * num_clients)))
    selected = random.sample(range(num_clients), m)

    # Pre-aggregation global vector for drift measurement
    with torch.no_grad():
        theta_global_vec = vectorize_params(global_model).to(device)

    local_states = []
    weights = []
    drifts = []

    for idx in selected:
        client_model = copy.deepcopy(global_model).to(device)
        # Local train
        local_sd = local_train(client_model, client_loaders[idx], device, K_local_epochs, lr,
                               weight_decay=weight_decay, momentum=momentum)
        local_states.append(local_sd)
        weights.append(client_sizes[idx])

        # Drift: ||theta_i^K - theta_global||
        client_model.load_state_dict(local_sd, strict=True)
        with torch.no_grad():
            theta_local_vec = vectorize_params(client_model).to(device)
            drift = torch.norm(theta_local_vec - theta_global_vec, p=2).item()
            drifts.append(drift)

    # Weighted aggregation by data size
    total_size = float(sum(weights))
    w_norm = [w / total_size for w in weights]
    new_global_sd = average_state_dicts(local_states, w_norm)

    avg_drift = float(sum(drifts) / len(drifts)) if len(drifts) > 0 else 0.0
    return new_global_sd, avg_drift, len(selected)

# -----------------------------
# Experiment runners
# -----------------------------
def run_experiment_vary_K(
    Ks=(1, 5, 10, 20),
    rounds=20,
    num_clients=10,
    lr=0.05,
    batch_size=64,
    sample_fraction=1.0,    # full participation for this experiment
    momentum=0.0,
    weight_decay=0.0,
    seed=42
):
    set_seed(seed)
    device = get_device()
    print(f"[Vary-K] Device: {device}")

    client_loaders, client_sizes, test_loader = get_cifar10_loaders_iid(
        num_clients=num_clients, batch_size=batch_size
    )

    results = defaultdict(list)  # key: K, value: list of dicts per round
    for K in Ks:
        print(f"\n[Vary-K] Running K={K}")
        global_model = SmallCIFAR10CNN().to(device)
        criterion = nn.CrossEntropyLoss()

        # Evaluate initial
        _, acc0 = evaluate(global_model, test_loader, device, criterion)
        print(f"[Round 0] K={K} | Acc={acc0:.4f}")

        comm_cost_cum = 0
        for t in range(1, rounds + 1):
            new_sd, avg_drift, num_participants = fedavg_round(
                global_model, client_loaders, client_sizes, device,
                K_local_epochs=K, lr=lr, sample_fraction=sample_fraction,
                momentum=momentum, weight_decay=weight_decay
            )
            global_model.load_state_dict(new_sd, strict=True)

            # Communication cost (count client updates transmitted this round)
            comm_cost_cum += num_participants

            # Evaluate
            loss, acc = evaluate(global_model, test_loader, device, criterion)
            print(f"[Round {t:02d}] K={K} | Acc={acc:.4f} | Drift={avg_drift:.4f} | Clients={num_participants}")

            results[K].append({
                "round": t,
                "acc": acc,
                "loss": loss,
                "avg_drift": avg_drift,
                "clients": num_participants,
                "comm_cost_cum": comm_cost_cum
            })
    return results

def run_experiment_vary_fraction(
    fractions=(1.0, 0.5, 0.2),
    rounds=20,
    num_clients=10,
    K=5,                    # fixed K
    lr=0.05,
    batch_size=64,
    momentum=0.0,
    weight_decay=0.0,
    seed=123
):
    set_seed(seed)
    device = get_device()
    print(f"[Vary-f] Device: {device}")

    client_loaders, client_sizes, test_loader = get_cifar10_loaders_iid(
        num_clients=num_clients, batch_size=batch_size
    )

    results = defaultdict(list)  # key: fraction, value: list of dicts per round
    for f in fractions:
        print(f"\n[Vary-f] Running fraction={f}")
        global_model = SmallCIFAR10CNN().to(device)
        criterion = nn.CrossEntropyLoss()

        # Evaluate initial
        _, acc0 = evaluate(global_model, test_loader, device, criterion)
        print(f"[Round 0] f={f} | Acc={acc0:.4f}")

        comm_cost_cum = 0
        for t in range(1, rounds + 1):
            new_sd, avg_drift, num_participants = fedavg_round(
                global_model, client_loaders, client_sizes, device,
                K_local_epochs=K, lr=lr, sample_fraction=f,
                momentum=momentum, weight_decay=weight_decay
            )
            global_model.load_state_dict(new_sd, strict=True)

            # Communication cost
            comm_cost_cum += num_participants

            # Evaluate
            loss, acc = evaluate(global_model, test_loader, device, criterion)
            print(f"[Round {t:02d}] f={f} | Acc={acc:.4f} | Drift={avg_drift:.4f} | Clients={num_participants}")

            results[f].append({
                "round": t,
                "acc": acc,
                "loss": loss,
                "avg_drift": avg_drift,
                "clients": num_participants,
                "comm_cost_cum": comm_cost_cum
            })
    return results

# -----------------------------
# Plotting (optional)
# -----------------------------
def try_plot(results_dict, title, xkey="round", ykey="acc", label_prefix=""):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        for k, series in results_dict.items():
            xs = [d[xkey] for d in series]
            ys = [d[ykey] for d in series]
            plt.plot(xs, ys, label=f"{label_prefix}{k}")
        plt.xlabel(xkey.capitalize())
        plt.ylabel(ykey.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"(Plot skipped) {e}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 2: FedAvg with K and client sampling experiments (CIFAR-10).")
    parser.add_argument("--rounds", type=int, default=20, help="Communication rounds per experiment.")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of federated clients.")
    parser.add_argument("--batch_size", type=int, default=64, help="Local batch size.")
    parser.add_argument("--lr", type=float, default=0.05, help="Local learning rate.")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true", help="Disable matplotlib plots.")
    args = parser.parse_args()

    # --- Experiment 1: Vary K with full participation ---
    Ks = (1, 5, 10, 20)
    res_varyK = run_experiment_vary_K(
        Ks=Ks,
        rounds=args.rounds,
        num_clients=args.num_clients,
        lr=args.lr,
        batch_size=args.batch_size,
        sample_fraction=1.0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed
    )

    # --- Experiment 2: Vary sampling fraction with fixed K ---
    fractions = (1.0, 0.5, 0.2)
    res_varyF = run_experiment_vary_fraction(
        fractions=fractions,
        rounds=args.rounds,
        num_clients=args.num_clients,
        K=5,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed + 1  # slight change for variety
    )

    # Optional: quick plots
    if not args.no_plots:
        try_plot(res_varyK, title="Accuracy vs Rounds (vary K, full participation)", ykey="acc", label_prefix="K=")
        try_plot(res_varyK, title="Drift vs Rounds (vary K, full participation)", ykey="avg_drift", label_prefix="K=")
        try_plot(res_varyK, title="Comm Cost (cumulative) vs Rounds (vary K)", ykey="comm_cost_cum", label_prefix="K=")

        try_plot(res_varyF, title="Accuracy vs Rounds (vary fraction, K=5)", ykey="acc", label_prefix="f=")
        try_plot(res_varyF, title="Drift vs Rounds (vary fraction, K=5)", ykey="avg_drift", label_prefix="f=")
        try_plot(res_varyF, title="Comm Cost (cumulative) vs Rounds (vary fraction, K=5)", ykey="comm_cost_cum", label_prefix="f=")

    # Print final summary
    print("\n=== Summary: Final Round Metrics ===")
    last_round = args.rounds
    for K in Ks:
        final = res_varyK[K][-1]
        print(f"[Vary-K] K={K} -> Acc={final['acc']:.4f} | Drift={final['avg_drift']:.4f} | CommCost={final['comm_cost_cum']}")
    for f in fractions:
        final = res_varyF[f][-1]
        print(f"[Vary-f] f={f} -> Acc={final['acc']:.4f} | Drift={final['avg_drift']:.4f} | CommCost={final['comm_cost_cum']}")

if __name__ == "__main__":
    main()