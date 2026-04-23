#!/usr/bin/env python3
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# -----------------------------
# Config (edit these if needed)
# -----------------------------
K = 1                     # number of affine pieces
tau_schedule = [1e-1, 1e-2, 1e-3, 1e-4]  # log-sum-exp temps (anneal)
lr = 1e-3
weight_decay = 1e-6
num_epochs = 1600
batch_size = 512
div_lambda = 10             # diversity penalty strength (0 to disable)
div_rho = 0.3                  # target upper bound on cosine similarity
row_normalize = False          # set True if you see instability

snapshot_dir = f'maxaffine_K{K}_wd_{weight_decay:.0e}_0_25eV_oneB'
os.makedirs(snapshot_dir, exist_ok=True)

# -----------------------------
# Load data (as in your script)
# -----------------------------
with open('OH_elements.json', 'r') as file:
    OH_elements = json.load(file)

with open('formation_energies_valid_0_25eV.json', 'r') as file:
    formation_energies_valid = json.load(file)

with open('p_dict_oh.json', 'r') as file:
    p_dict_oh = json.load(file)

features_tensor = torch.load('features_tensor_0_25eV.pt')  # list of tensors

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Stack features -> [N, d]
X_all = torch.stack(features_tensor)           # float32 likely; keep as-is
d = X_all.shape[-1]

# Scale target to [0,1] (same as your script)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_all = torch.tensor(
    scaler_y.fit_transform(np.array(formation_energies_valid).reshape(-1, 1)),
    dtype=torch.float32
).squeeze(-1)   # [N]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=1278
)

# Tensors on device, correct shape
X_train = X_train.to(device).float()
X_test  = X_test.to(device).float()
y_train = y_train.to(device).float().view(-1)     # [N]
y_test  = y_test.to(device).float().view(-1)      # [N]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# -----------------------------------
# Max-Affine model (log-sum-exp train)
# -----------------------------------
class SoftMaxAffine(nn.Module):
    """
    g_tau(x) = tau * logsumexp_j( (w_j^T x + b_j) / tau )
    As tau -> 0, this approaches max_j w_j^T x + b_j.
    """
    def __init__(self, in_dim, K, tau=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(K, in_dim) / (in_dim ** 0.5))  # [K,d]
        self.b = nn.Parameter(torch.zeros(1))                             # [1]
        self.tau = tau

    def forward(self, x):
        # x: [N,d]
        z = x @ self.W.t() + self.b[:, None]            # [N,K]
        y = self.tau * torch.logsumexp(z / self.tau, dim=1)  # [N]
        return y, z

    @torch.no_grad()
    def hard(self, x):
        z = x @ self.W.t() + self.b[:, None]
        y = z.max(dim=1).values
        return y, z

def row_normalize_(W, eps=1e-12):
    with torch.no_grad():
        norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
        W.div_(norms)

def diversity_penalty(W, rho=0.2):
    # Encourage pairwise cosines <= rho
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-12)  # [K,d]
    C = (Wn @ Wn.t())                                # [K,K]
    tri = torch.triu(C, diagonal=1)
    return torch.mean(torch.clamp(tri - rho, min=0.0) ** 2)

def rmse_torch(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))

# Init model/opt
model = SoftMaxAffine(d, K, tau=tau_schedule[0]).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# -----------------------------
# Training loop with annealing
# -----------------------------
stages = len(tau_schedule)
steps_per_stage = max(1, num_epochs // stages)

for s, tau in enumerate(tau_schedule, 1):
    model.tau = tau
    for step in range(steps_per_stage):
        for Xb, yb in train_loader:
            model.train()
            optimizer.zero_grad()

            yhat, z = model(Xb)                      # [N]
            mse = torch.mean((yhat - yb) ** 2)

            div = div_lambda * diversity_penalty(model.W, rho=div_rho) if div_lambda > 0 else 0.0
            loss = mse + div

            loss.backward()
            optimizer.step()
            if row_normalize:
                row_normalize_(model.W)
        print(f"    Stage {s}/{stages} | tau={tau} | Step {step+1}/{steps_per_stage} | Loss: {loss.item():.6f} | MSE: {mse.item():.6f} | Div: {div.item() if div_lambda > 0 else 0.0:.6f}")

    # quick stage summary (hard max)
    model.eval()
    with torch.no_grad():
        yhat_tr_std, _ = model.hard(X_train)
        yhat_te_std, _ = model.hard(X_test)
        tr_rmse_std = rmse_torch(yhat_tr_std, y_train).item()
        te_rmse_std = rmse_torch(yhat_te_std, y_test).item()
    print(f"[Stage {s}/{stages} | tau={tau}] RMSE_std: train={tr_rmse_std:.6f}, test={te_rmse_std:.6f}")

# --------------------------------
# Final evaluation (HARD MAX ONLY)
# --------------------------------
model.eval()
with torch.no_grad():
    yhat_tr_std, ztr = model.hard(X_train)
    yhat_te_std, zte = model.hard(X_test)

# Inverse-transform y to original units for metrics/plots
y_train_np = y_train.detach().cpu().numpy().reshape(-1, 1)
y_test_np  = y_test.detach().cpu().numpy().reshape(-1, 1)
yhat_tr_std_np = yhat_tr_std.detach().cpu().numpy().reshape(-1, 1)
yhat_te_std_np = yhat_te_std.detach().cpu().numpy().reshape(-1, 1)

y_train_inv = scaler_y.inverse_transform(y_train_np).reshape(-1)
y_test_inv  = scaler_y.inverse_transform(y_test_np).reshape(-1)
yhat_tr_inv = scaler_y.inverse_transform(yhat_tr_std_np).reshape(-1)
yhat_te_inv = scaler_y.inverse_transform(yhat_te_std_np).reshape(-1)

# Metrics
def r2_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - (ss_res / (ss_tot + 1e-12))

mae_train = mean_absolute_error(y_train_inv, yhat_tr_inv)
mae_test  = mean_absolute_error(y_test_inv,  yhat_te_inv)
rmse_train = np.sqrt(mean_squared_error(y_train_inv, yhat_tr_inv))
rmse_test  = np.sqrt(mean_squared_error(y_test_inv,  yhat_te_inv))
r2_train = r2_numpy(y_train_inv, yhat_tr_inv)
r2_test  = r2_numpy(y_test_inv,  yhat_te_inv)

print("\n=== Results (original target units) ===")
print(f"Train RMSE: {rmse_train:.6f}  |  MAE: {mae_train:.6f}  |  R^2: {r2_train:.6f}")
print(f"Test  RMSE: {rmse_test:.6f}   |  MAE: {mae_test:.6f}   |  R^2: {r2_test:.6f}")

# -----------------------------
# Save planes and “activeness”
# -----------------------------
# Active planes = planes that are argmax for at least one point (train/test)
owners_tr = torch.argmax(ztr, dim=1).detach().cpu().numpy()
owners_te = torch.argmax(zte, dim=1).detach().cpu().numpy()
active_tr = np.unique(owners_tr)
active_te = np.unique(owners_te)
active_any = np.unique(np.concatenate([active_tr, active_te]))
print(f"\nActive planes: train={len(active_tr)}/{K}, test={len(active_te)}/{K}, any={len(active_any)}/{K}")

# Save standardization params for y (so plots/metrics reproducible)
scaler_params = {
    'min_': scaler_y.min_.tolist(),
    'scale_': scaler_y.scale_.tolist(),
    'data_min_': scaler_y.data_min_.tolist(),
    'data_max_': scaler_y.data_max_.tolist(),
    'data_range_': scaler_y.data_range_.tolist(),
    'feature_range': scaler_y.feature_range
}
with open(os.path.join(snapshot_dir, 'scaler_y.json'), 'w') as f:
    json.dump(scaler_params, f, indent=2)

# Save model parameters (in the same input units your model sees)
torch.save({'W': model.W.detach().cpu(), 'b': model.b.detach().cpu()},
           os.path.join(snapshot_dir, f'maxaffine_K{K}_0_25eV_oneB.pt'))

# Also dump a CSV for quick inspection
import pandas as pd
W_np = model.W.detach().cpu().numpy()
b_np = model.b.detach().cpu().numpy()
planes = pd.DataFrame(np.hstack([W_np, b_np*np.ones((K,1))]),
                      columns=[f"w_{i}" for i in range(d)] + ["b"])
planes["active_train"] = 0
planes.loc[active_tr, "active_train"] = 1
planes["active_any"] = 0
planes.loc[active_any, "active_any"] = 1
planes.to_csv(os.path.join(snapshot_dir, f'planes_K{K}.csv'), index=False)
print(f"Saved planes to {snapshot_dir}/planes_K{K}.csv")

# -----------------------------
# Plots (train & test)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_train_inv, yhat_tr_inv, edgecolor='k', alpha=0.7, label='Training Data')
lo, hi = min(y_train_inv.min(), yhat_tr_inv.min()), max(y_train_inv.max(), yhat_tr_inv.max())
plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Formation Energies (eV)', fontsize=16)
plt.ylabel('Predicted Formation Energies (eV)', fontsize=16)
plt.legend()
plt.xticks(fontsize=14); plt.yticks(fontsize=14); plt.grid(True)
plt.text(0.65, 0.05, f'RMSE: {rmse_train:.4f}\nMAE: {mae_train:.4f}\nR^2: {r2_train:.4f}',
         transform=plt.gca().transAxes, fontsize=12, va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(snapshot_dir, 'final_training_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_inv, yhat_te_inv, color='green', edgecolor='k', alpha=0.7, label='Testing Data')
lo, hi = min(y_test_inv.min(), yhat_te_inv.min()), max(y_test_inv.max(), yhat_te_inv.max())
plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Formation Energies (eV)', fontsize=16)
plt.ylabel('Predicted Formation Energies (eV)', fontsize=16)
plt.legend()
plt.xticks(fontsize=14); plt.yticks(fontsize=14); plt.grid(True)
plt.text(0.65, 0.05, f'RMSE: {rmse_test:.4f}\nMAE: {mae_test:.4f}\nR^2: {r2_test:.4f}',
         transform=plt.gca().transAxes, fontsize=12, va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(snapshot_dir, 'final_testing_plot.png'))
plt.close()

print(f"\nFigures & artifacts saved to: {snapshot_dir}")
