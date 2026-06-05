import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv('../data/behavioral_data.csv')
feature_cols = ['dwell_time', 'flight_time', 'press_speed', 
                'mouse_speed', 'click_interval']
X = df[feature_cols]
y = df['label']

# Train on normal data only
X_normal = X[y == 0]
model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
model.fit(X_normal)

# Test
preds = model.predict(X)
preds_binary = (preds == -1).astype(int)

print('Results:')
print(classification_report(y, preds_binary, 
      target_names=['Normal', 'Anomaly']))

# Save model
with open('../models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved to models/isolation_forest.pkl ✅')