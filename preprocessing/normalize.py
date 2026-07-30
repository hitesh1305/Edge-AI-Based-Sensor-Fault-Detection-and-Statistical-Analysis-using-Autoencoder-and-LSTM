from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption"
]


def normalize_features(df):

    scaler = MinMaxScaler()

    df[FEATURE_COLUMNS] = scaler.fit_transform(
        df[FEATURE_COLUMNS]
    )

    return df, scaler