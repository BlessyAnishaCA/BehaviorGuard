import pandas as pd
import numpy as np

def build_features(df):
    features = pd.DataFrame()

    # Amount — normalize it
    features['amount_log'] = np.log1p(df['Amount'])

    # Time — normalize it
    features['time_norm'] = df['Time'] / df['Time'].max()

    # V1 to V28 features
    v_cols = [f'V{i}' for i in range(1, 29)]
    for col in v_cols:
        features[col] = df[col]

    # Fill missing values
    features = features.fillna(0)

    return features

if __name__ == '__main__':
    print("Loading dataset...")
    df = pd.read_csv('../data/creditcard.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Normal cases: {(df['Class']==0).sum()}")

    X = build_features(df)
    y = df['Class']

    X.to_csv('../data/features.csv', index=False)
    y.to_csv('../data/labels.csv', index=False)
    print("Saved features.csv and labels.csv ✅")