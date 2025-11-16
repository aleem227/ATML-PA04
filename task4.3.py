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
import numpy as np

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

def harmonize_gradients(grad_list, eps: float = 1e-12, cos_threshold: float = -0.1):
    """
    FedGH (milder version):

    - Work with *flattened* client updates g_i.
    - For each pair (j, i) with j < i:
        * compute cosine similarity cos(g_i, g_j).
        * if cos < cos_threshold (strongly conflicting),
          project g_i onto the orthogonal complement of g_j:

              g_i <- g_i - (g_i · g_j / ||g_j||^2) g_j

      Only g_i is modified (one-sided projection); g_j is kept as an anchor.

    This makes GH less aggressive and closer to FedAvg, which helps stability.
    """
    if len(grad_list) <= 1:
        return grad_list

    grads = [g.clone() for g in grad_list]
    m = len(grads)

    for i in range(m):
        for j in range(i):  # only pairs (j, i) with j < i, modify g_i only
            gi = grads[i]
            gj = grads[j]

            gi_norm = gi.norm()
            gj_norm = gj.norm()
            if gi_norm.item() < eps or gj_norm.item() < eps:
                # Ignore near-zero vectors to avoid huge projections
                continue

            dot = torch.dot(gi, gj)
            cos_sim = dot / (gi_norm * gj_norm + eps)

            # Only treat *strong* conflicts as needing harmonization
            if cos_sim.item() < cos_threshold:
                # Project g_i onto orthogonal complement of g_j:
                # g_i <- g_i - (g_i · g_j / ||g_j||^2) g_j
                proj_coeff = dot / (gj_norm.pow(2) + eps)
                grads[i] = gi - proj_coeff * gj

    return grads

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
# Non-IID via Dirichlet
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

        # Fix rounding to keep total exactly n_c
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
    """
    Non-IID CIFAR-10 loaders using Dirichlet(alpha) label partition.

    Returns:
      client_loaders: list of DataLoader per client
      client_sizes: list of dataset sizes per client
      test_loader: global CIFAR-10 test loader
    """
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    labels = np.array(trainset.targets)
    client_idx_lists = dirichlet_split_indices(labels, num_clients, alpha, seed)

    client_subsets = [Subset(trainset, idxs) for idxs in client_idx_lists]
    client_loaders = [DataLoader(s, batch_size=batch_size, shuffle=True, drop_last=False)
                      for s in client_subsets]
    client_sizes = [len(s) for s in client_subsets]

    test_loader = DataLoader(testset, batch_size=512, shuffle=False)
    return client_loaders, client_sizes, test_loader

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

def local_train_sam(model, loader, device, epochs, lr, rho,
                    weight_decay=0.0, momentum=0.0):
    """
    Local training with SAM (Sharpness-Aware Minimization).

    For each batch:
      1) Compute gradients at current weights w.
      2) Perturb weights to w_adv = w + rho * g / ||g||.
      3) Compute gradients at w_adv.
      4) Restore w, then apply optimizer.step() using grad at w_adv.
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 1) Ascent step: grad at current weights
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Compute global grad norm
            grad_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += p.grad.detach().pow(2).sum()
            grad_norm = (grad_norm_sq + 1e-12).sqrt()

            # 2) Perturb weights: w_adv = w + rho * g / ||g||
            #    Keep a per-param copy of the perturbation so we can undo it.
            e_ws = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        e_ws.append(None)
                        continue
                    e_w = rho * p.grad / grad_norm
                    p.add_(e_w)
                    e_ws.append(e_w)

            # 3) Descent step: gradients at perturbed weights w_adv
            optimizer.zero_grad()
            logits_adv = model(x)
            loss_adv = criterion(logits_adv, y)
            loss_adv.backward()

            # 4) Restore original weights and apply optimizer step
            with torch.no_grad():
                for p, e_w in zip(model.parameters(), e_ws):
                    if e_w is None:
                        continue
                    p.sub_(e_w)

            optimizer.step()

    return model.state_dict()

# -----------------------------
# FedAvg round
# -----------------------------
def fedavg_round(global_model,
                 client_loaders,
                 client_sizes,
                 device,
                 K_local_epochs,
                 lr,
                 sample_fraction=1.0,
                 momentum=0.0,
                 weight_decay=0.0,
                 use_gh: bool = False,
                 use_sam: bool = False,
                 sam_rho: float = 0.05):
    """
    Perform one communication round:

      1) Sample a subset of clients (fraction).
      2) Send global to each sampled client.
      3) Each client trains K epochs locally (SGD or SAM).
      4) Optionally harmonize updates (FedGH) before aggregation.
      5) Aggregate weighted by data size.

    Returns:
      new_global_state_dict, avg_client_drift, num_participants
    """
    num_clients = len(client_loaders)
    m = max(1, int(math.ceil(sample_fraction * num_clients)))
    selected = random.sample(range(num_clients), m)

    # Pre-aggregation global vector for drift measurement
    with torch.no_grad():
        theta_global_vec = vectorize_params(global_model).to(device)

    # Keep a copy of the starting global state for delta computation (FedGH)
    global_start_sd = copy.deepcopy(global_model.state_dict())

    local_states = []
    weights = []
    drifts = []

    for idx in selected:
        client_model = copy.deepcopy(global_model).to(device)

        # Local training: FedAvg (SGD) or FedSAM
        if use_sam:
            local_sd = local_train_sam(
                client_model,
                client_loaders[idx],
                device,
                epochs=K_local_epochs,
                lr=lr,
                rho=sam_rho,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            local_sd = local_train(
                client_model,
                client_loaders[idx],
                device,
                epochs=K_local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )

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

    # --- FedGH: harmonize client updates on the server before averaging ---
    if use_gh and len(local_states) > 1:
        # Flatten per-client deltas g_i = theta_i^K - theta_global_start
        param_keys = list(global_start_sd.keys())
        shapes = [global_start_sd[k].shape for k in param_keys]
        numels = [global_start_sd[k].numel() for k in param_keys]

        flat_updates = []
        for sd in local_states:
            chunks = []
            for k in param_keys:
                delta = (sd[k] - global_start_sd[k]).float()
                chunks.append(delta.view(-1))
            flat_updates.append(torch.cat(chunks))

        # Harmonize gradients (FedGH)
        harmonized = harmonize_gradients(flat_updates)

        # Map harmonized flat updates back to state_dicts
        adj_local_states = []
        for g_vec in harmonized:
            offset = 0
            new_sd = {}
            for k, shape, n in zip(param_keys, shapes, numels):
                chunk = g_vec[offset:offset + n].view(shape)
                offset += n
                new_sd[k] = global_start_sd[k] + chunk
            adj_local_states.append(new_sd)

        agg_states = adj_local_states
    else:
        # Plain FedAvg aggregation
        agg_states = local_states

    new_global_sd = average_state_dicts(agg_states, w_norm)
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
    seed=42,
    use_gh: bool = False,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    non_iid: bool = False,
    dirichlet_alpha: float = 0.1,
):

    set_seed(seed)
    device = get_device()
    print(f"[Vary-K] Device: {device}")



    if non_iid:
        client_loaders, client_sizes, test_loader = get_cifar10_dirichlet_loaders(
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            batch_size=batch_size,
            data_root="./data",
            seed=seed,
        )
        print(f"[Vary-K] Using non-IID Dirichlet split (alpha={dirichlet_alpha})")
    else:
        client_loaders, client_sizes, test_loader = get_cifar10_loaders_iid(
            num_clients=num_clients,
            batch_size=batch_size,
            data_root="./data",
        )
        print("[Vary-K] Using IID equal split")


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
                global_model,
                client_loaders,
                client_sizes,
                device,
                K_local_epochs=K,
                lr=lr,
                sample_fraction=sample_fraction,
                momentum=momentum,
                weight_decay=weight_decay,
                use_gh=use_gh,
                use_sam=use_sam,
                sam_rho=sam_rho,
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
    seed=123,
    use_gh: bool = False,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    non_iid: bool = False,
    dirichlet_alpha: float = 0.1,
):

    set_seed(seed)
    device = get_device()
    print(f"[Vary-f] Device: {device}")

    if non_iid:
        client_loaders, client_sizes, test_loader = get_cifar10_dirichlet_loaders(
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            batch_size=batch_size,
            data_root="./data",
            seed=seed,
        )
        print(f"[Vary-f] Using non-IID Dirichlet split (alpha={dirichlet_alpha})")
    else:
        client_loaders, client_sizes, test_loader = get_cifar10_loaders_iid(
            num_clients=num_clients,
            batch_size=batch_size,
            data_root="./data",
        )
        print("[Vary-f] Using IID equal split")

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
                global_model,
                client_loaders,
                client_sizes,
                device,
                K_local_epochs=K,
                lr=lr,
                sample_fraction=f,
                momentum=momentum,
                weight_decay=weight_decay,
                use_gh=use_gh,
                use_sam=use_sam,
                sam_rho=sam_rho,
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

        # Non-IID partitioning (Task 4)
    parser.add_argument("--non_iid", action="store_true",
                        help="Use Dirichlet non-IID partition for CIFAR-10.")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1,
                        help="Dirichlet concentration alpha for non-IID split (smaller = more skewed).")


    parser.add_argument("--use_gh", action="store_true",
                        help="Enable Gradient Harmonization (FedGH) on the server.")
    parser.add_argument("--use_sam", action="store_true",
                        help="Enable SAM on clients (FedSAM).")
    parser.add_argument("--sam_rho", type=float, default=0.05,
                        help="SAM radius rho for FedSAM.")
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
        seed=args.seed,
        use_gh=args.use_gh,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        non_iid=args.non_iid,
        dirichlet_alpha=args.dirichlet_alpha,
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
        seed=args.seed + 1,  # slight change for variety
        use_gh=args.use_gh,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        non_iid=args.non_iid,
        dirichlet_alpha=args.dirichlet_alpha,
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


#python task2.py --non_iid --dirichlet_alpha 0.1 --use_gh

#python task2.py --non_iid --dirichlet_alpha 0.1 --use_sam --sam_rho 0.05
