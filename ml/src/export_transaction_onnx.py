import torch
from train_transaction import Autoencoder

print("Loading autoencoder model...")
model = Autoencoder()
model.load_state_dict(torch.load('../models/autoencoder.pth'))
model.eval()

print("Converting to ONNX...")
dummy_input = torch.randn(1, 5)
torch.onnx.export(
    model, dummy_input,
    '../models/transaction.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}
)
print("transaction.onnx saved to models/ ✅")