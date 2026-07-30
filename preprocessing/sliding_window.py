import numpy as np

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption"
]


def create_sequences(df, window_size=20):

    data = df[FEATURE_COLUMNS].values

    X = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])

    return np.array(X)