import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Load features ──────────────────────────────────────────────
print("Loading features...")
X = pd.read_csv('../data/features.csv').values.astype('float32')
y = pd.read_csv('../data/labels.csv').values.ravel()

# Train only on normal transactions
X_normal = X[y == 0]
print(f"Training on {len(X_normal)} normal transactions...")

X_tensor = torch.tensor(X_normal)
dataset   = TensorDataset(X_tensor, X_tensor)
loader    = DataLoader(dataset, batch_size=64, shuffle=True)

# ── Define Autoencoder ─────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Train ──────────────────────────────────────────────────────
input_dim = X.shape[1]
model     = Autoencoder(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training started...")
for epoch in range(30):
    total_loss = 0
    for batch_x, _ in loader:
        output = model(batch_x)
        loss   = criterion(output, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30  Loss: {total_loss/len(loader):.4f}")

# ── Save ───────────────────────────────────────────────────────
torch.save(model.state_dict(), '../models/autoencoder.pth')
print("Model saved to models/autoencoder.pth ✅")