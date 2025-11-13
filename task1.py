import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

# =========================================================
# 1. DEVICE SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# 2. PARAMETERS
# =========================================================
num_clients = 3          # number of clients
lr = 0.01                # learning rate
epochs = 10            # number of rounds
batch_size = 512
torch.manual_seed(42)

# =========================================================
# 3. DATASET (IID SPLIT)
# =========================================================
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Equal IID partitions
client_size = len(dataset) // num_clients
clients = random_split(dataset, [client_size]*num_clients)

# Combine all client data for centralized training
central_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# =========================================================
# 4. SIMPLE MODEL
# =========================================================
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================================================
# 5. LOSS FUNCTION
# =========================================================
criterion = nn.CrossEntropyLoss()

# =========================================================
# 6. INITIALIZE MODELS
# =========================================================
global_model_central = SmallNet().to(device)
global_model_fed = copy.deepcopy(global_model_central).to(device)

optimizer_central = optim.SGD(global_model_central.parameters(), lr=lr)

# =========================================================
# 7. HELPER FUNCTIONS
# =========================================================
def compute_fullbatch_gradients(model, data_loader):
    """Compute gradients over full dataset for one client."""
    model.zero_grad()
    loss_total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss_total += criterion(preds, y) * len(x)
    loss_total = loss_total / len(data_loader.dataset)
    loss_total.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    return grads

def apply_gradients(model, grads, lr):
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p -= lr * g

# =========================================================
# 8. TRAINING LOOP
# =========================================================
for epoch in range(epochs):

    # -------------------------------
    # Centralized SGD Step
    # -------------------------------
    for x, y in central_loader:
        x, y = x.to(device), y.to(device)
        optimizer_central.zero_grad()
        preds = global_model_central(x)
        loss_central = criterion(preds, y)
        loss_central.backward()
        optimizer_central.step()
        break  # only one full-batch step per epoch

    # -------------------------------
    # Federated SGD Step (K=1)
    # -------------------------------
    grads_sum = None
    total_data = sum(len(c) for c in clients)

    for client_data in clients:
        loader = DataLoader(client_data, batch_size=len(client_data), shuffle=False)
        local_model = copy.deepcopy(global_model_fed).to(device)
        grads = compute_fullbatch_gradients(local_model, loader)

        # Weighted aggregation (Ni/N)
        weight = len(client_data) / total_data
        grads = [g * weight for g in grads]

        if grads_sum is None:
            grads_sum = grads
        else:
            grads_sum = [gs + g for gs, g in zip(grads_sum, grads)]

    # Apply aggregated gradients
    apply_gradients(global_model_fed, grads_sum, lr)

    # -------------------------------
    # Compare parameter difference
    # -------------------------------
    diff_norm = 0.0
    with torch.no_grad():
        for p1, p2 in zip(global_model_central.parameters(), global_model_fed.parameters()):
            diff_norm += torch.norm(p1 - p2).item()

    print(f"Round {epoch+1:02d} | Δθ norm: {diff_norm:.8f}")

# =========================================================
# 9. FINAL VERIFICATION (OPTIONAL)
# =========================================================
# Evaluate both on same validation split
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

def evaluate(model, loader):
    correct, total, loss_sum = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(x)
    return loss_sum/total, correct/total

loss_c, acc_c = evaluate(global_model_central, test_loader)
loss_f, acc_f = evaluate(global_model_fed, test_loader)

print("\n=== Final Comparison ===")
print(f"Centralized  | Loss: {loss_c:.4f}, Acc: {acc_c*100:.2f}%")
print(f"FedSGD       | Loss: {loss_f:.4f}, Acc: {acc_f*100:.2f}%")
