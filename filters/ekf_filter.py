from filterpy.kalman import KalmanFilter
import numpy as np

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption"
]


def apply_ekf(df):
    """
    Apply a 1D Kalman Filter to each sensor feature.
    Returns a new dataframe with smoothed values.
    """

    filtered_df = df.copy()

    for column in FEATURE_COLUMNS:

        kf = KalmanFilter(dim_x=1, dim_z=1)

        # State transition
        kf.F = np.array([[1]])

        # Measurement function
        kf.H = np.array([[1]])

        # Initial state
        kf.x = np.array([[df[column].iloc[0]]])

        # Covariance matrices
        kf.P *= 10.0          # Initial uncertainty
        kf.R = np.array([[0.01]])   # Measurement noise
        kf.Q = np.array([[0.001]])  # Process noise

        filtered_values = []

        for value in df[column]:

            kf.predict()
            
            kf.update(np.array([[value]]))

            filtered_values.append(kf.x[0, 0])

        filtered_df[column] = filtered_values

    return filtered_df