import pickle
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load trained model
print("Loading isolation forest model...")
with open('../models/isolation_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Export to ONNX
print("Converting to ONNX...")
initial_type = [('float_input', FloatTensorType([None, 5]))]
onnx_model = convert_sklearn(model, initial_types=initial_type,
             target_opset={'': 18, 'ai.onnx.ml': 3})

with open('../models/behavior.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("behavior.onnx saved to models/ ✅")