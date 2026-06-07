import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
import pickle

print("Loading data and model...")
df = pd.read_csv('../data/behavioral_data.csv')
feature_cols = ['dwell_time', 'flight_time', 'press_speed',
                'mouse_speed', 'click_interval']
X = df[feature_cols]
y = df['label']

with open('../models/isolation_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Predictions
preds_raw = model.predict(X)
preds     = (preds_raw == -1).astype(int)
scores    = -model.score_samples(X)

print("Generating charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('BehaviorGuard — ML Model Evaluation', 
             fontsize=16, fontweight='bold', color='#065F46')

# ── Chart 1 — Confusion Matrix ──────────────────────────────
cm = confusion_matrix(y, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=axes[0], linewidths=1)
axes[0].set_title('Confusion Matrix\nBehavioral Anomaly Model',
                  fontsize=13, fontweight='bold', color='#065F46')
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=11)

# Add accuracy text
accuracy = (preds == y).mean()
axes[0].text(0.5, -0.15, f'Accuracy: {accuracy:.1%}',
             transform=axes[0].transAxes,
             ha='center', fontsize=12, fontweight='bold',
             color='#065F46')

# ── Chart 2 — ROC Curve ─────────────────────────────────────
fpr, tpr, _ = roc_curve(y, scores)
auc = roc_auc_score(y, scores)

axes[1].plot(fpr, tpr, color='#065F46', lw=2.5, label='Isolation Forest (AUC = ' + str(round(auc, 2)) + ')')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#065F46')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].set_title('ROC Curve\nBehavioral Anomaly Model',
                  fontsize=13, fontweight='bold', color='#065F46')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/evaluation_charts.png',
            dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()

print(f"AUC Score: {auc:.3f}")
print(f"Accuracy:  {accuracy:.1%}")
print()
print(classification_report(y, preds,
      target_names=['Normal', 'Anomaly']))
print("Charts saved to models/evaluation_charts.png ✅")