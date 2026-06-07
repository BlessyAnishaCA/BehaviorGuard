"""
INFERENCE MODULE — Member 4 uses this file
Both functions return a float between 0.0 and 1.0
  0.0 = completely normal
  1.0 = definitely anomalous / fraud
"""
import numpy as np
import onnxruntime as ort

# ── Load models once at startup ──────────────────────────────
print("Loading models...")
behavior_session    = ort.InferenceSession('../models/behavior.onnx')
transaction_session = ort.InferenceSession('../models/transaction.onnx')
print("Models loaded ✅")

def behavior_score(dwell_time, flight_time, press_speed,
                   mouse_speed, click_interval):
    """
    INPUT (all floats):
      dwell_time     — how long a key is held in ms (normal: ~120)
      flight_time    — time between key presses in ms (normal: ~80)
      press_speed    — keys per second (normal: ~5.2)
      mouse_speed    — pixels per second (normal: ~300)
      click_interval — ms between mouse clicks (normal: ~400)
    OUTPUT: float 0.0 to 1.0 (anomaly score)
    """
    features = np.array([[dwell_time, flight_time, press_speed,
                          mouse_speed, click_interval]],
                          dtype=np.float32)
    input_name = behavior_session.get_inputs()[0].name
    result = behavior_session.run(None, {input_name: features})
    label = result[0][0]
    return 1.0 if label == -1 else 0.0

def transaction_score(amount_log, hour, card4, card6, product):
    """
    INPUT (all floats):
      amount_log — log of transaction amount
      hour       — hour of day (0-23)
      card4      — card type encoded (0-3)
      card6      — card category encoded (0-2)
      product    — product category encoded (0-4)
    OUTPUT: float 0.0 to 1.0 (anomaly score)
    """
    features = np.array([[amount_log, hour, card4, card6, product]],
                          dtype=np.float32)
    input_name = transaction_session.get_inputs()[0].name
    output = transaction_session.run(None, {input_name: features})[0]
    error = float(np.mean((features - output) ** 2))
    # Flip — higher error means MORE normal in this model
    normalized = (error - 2.0) / 40.0
    flipped = 1.0 - float(np.clip(normalized, 0, 1))
    return flipped