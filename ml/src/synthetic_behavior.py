import numpy as np
import pandas as pd

def generate_normal_user(n=5000):
    return pd.DataFrame({
        'dwell_time':     np.random.normal(120, 15, n),
        'flight_time':    np.random.normal(80,  10, n),
        'press_speed':    np.random.normal(5.2, 0.4, n),
        'mouse_speed':    np.random.normal(300, 40, n),
        'click_interval': np.random.normal(400, 50, n),
        'label': 0
    })

def generate_attacker(n=1000):
    return pd.DataFrame({
        'dwell_time':     np.random.normal(80,  30, n),
        'flight_time':    np.random.normal(50,  25, n),
        'press_speed':    np.random.normal(7.0, 1.2, n),
        'mouse_speed':    np.random.normal(600, 120, n),
        'click_interval': np.random.normal(200, 80, n),
        'label': 1
    })

if __name__ == '__main__':
    normal   = generate_normal_user(5000)
    attacker = generate_attacker(1000)
    df = pd.concat([normal, attacker]).sample(frac=1).reset_index(drop=True)
    df.to_csv('../data/behavioral_data.csv', index=False)
    print(f'Generated {len(df)} behavioral samples')
    print(f'Normal: {(df.label==0).sum()}, Attacker: {(df.label==1).sum()}')