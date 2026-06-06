import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Load only 5 features ────────────────────────────────────
print("Loading features...")
df = pd.read_csv('../data/features.csv')

# Use only these 5 columns
feature_cols = ['amount_log', 'time_norm', 'V1', 'V2', 'V3']
X = df[feature_cols].values.astype('float32')
y = pd.read_csv('../data/labels.csv').values.ravel()

# Train only on normal transactions
X_normal = X[y == 0]
print(f"Training on {len(X_normal)} normal transactions...")

X_tensor = torch.tensor(X_normal)
dataset   = TensorDataset(X_tensor, X_tensor)
loader    = DataLoader(dataset, batch_size=64, shuffle=True)

# ── Define Autoencoder ──────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 5)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Train ───────────────────────────────────────────────────
model     = Autoencoder()
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

torch.save(model.state_dict(), '../models/autoencoder.pth')
print("Model saved to models/autoencoder.pth ✅")
