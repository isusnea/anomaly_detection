import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
#                   USER TUNING SECTION
#    Modify these values below to tune the membership functions
#    for OAL (Observed Activity Level) and Dist (Euclidean),
#    the moving average filter, and now the SVR model.
# ============================================================

FILTER_WINDOW = 5  # <-- Adjust the moving average filter window here

# ---- SVR TUNING PARAMETERS ----
SVR_KERNEL = 'rbf'
SVR_C = 1.0
SVR_EPSILON = 0.1

# ---- OAL THRESHOLDS ----
OAL_LOW_MAX = 0.0      # Full membership if OAL <= this
OAL_LOW_ZERO = 0.1     # Zero membership if OAL >= this (linear in between)

OAL_MED_ZERO_LEFT = 0.1   # OAL <= this => no membership for MED
OAL_MED_PEAK = 0.7          # Peak membership for MED
OAL_MED_ZERO_RIGHT = 1.5  # OAL >= this => no membership for MED

OAL_HIGH_ZERO_LEFT = 0.5  # OAL <= this => membership 0
OAL_HIGH_FULL = 1.5        # OAL >= this => membership 1.0 (linear in between)

# ---- DIST THRESHOLDS (Euclidean Distance) ----
DIST_LOW_MAX = 2.0     # Full membership if Dist <= this
DIST_LOW_ZERO = 10.0    # Zero membership if Dist >= this (linear in between)

DIST_MED_ZERO_LEFT = 5.0  # Dist <= this => no membership for MED
DIST_MED_PEAK = 10.0       # Dist at which MED is max
DIST_MED_ZERO_RIGHT = 15.0 # Dist >= this => no membership for MED

DIST_HIGH_ZERO_LEFT = 25.0 # Dist <= this => membership 0
DIST_HIGH_FULL = 35.0      # Dist >= this => membership 1.0 (linear in between)

# Singletons for rule outputs
PA_LOW_DIST_LOW  = 0.8
PA_LOW_DIST_MED  = 0.9
PA_LOW_DIST_HIGH = 1.0

PA_MED_DIST_LOW  = 0.3
PA_MED_DIST_MED  = 0.5
PA_MED_DIST_HIGH = 0.7

PA_OAL_HIGH      = 0.1

# ============================================================
#                END OF USER TUNING SECTION
# ============================================================

# ------------------------------------------------------------
# Fuzzy Membership Functions for OAL
# ------------------------------------------------------------
def oal_low(x):
    if x <= OAL_LOW_MAX:
        return 1.0
    elif x >= OAL_LOW_ZERO:
        return 0.0
    else:
        return 1.0 - (x - OAL_LOW_MAX) / (OAL_LOW_ZERO - OAL_LOW_MAX)

def oal_med(x):
    if x <= OAL_MED_ZERO_LEFT or x >= OAL_MED_ZERO_RIGHT:
        return 0.0
    elif x < OAL_MED_PEAK:
        return (x - OAL_MED_ZERO_LEFT) / (OAL_MED_PEAK - OAL_MED_ZERO_LEFT)
    else:
        return (OAL_MED_ZERO_RIGHT - x) / (OAL_MED_ZERO_RIGHT - OAL_MED_PEAK)

def oal_high(x):
    if x <= OAL_HIGH_ZERO_LEFT:
        return 0.0
    elif x >= OAL_HIGH_FULL:
        return 1.0
    else:
        return (x - OAL_HIGH_ZERO_LEFT) / (OAL_HIGH_FULL - OAL_HIGH_ZERO_LEFT)

# ------------------------------------------------------------
# Fuzzy Membership Functions for Dist (Euclidean Distance)
# ------------------------------------------------------------
def dist_low(d):
    if d <= DIST_LOW_MAX:
        return 1.0
    elif d >= DIST_LOW_ZERO:
        return 0.0
    else:
        return (DIST_LOW_ZERO - d) / (DIST_LOW_ZERO - DIST_LOW_MAX)

def dist_med(d):
    if d <= DIST_MED_ZERO_LEFT or d >= DIST_MED_ZERO_RIGHT:
        return 0.0
    elif d < DIST_MED_PEAK:
        return (d - DIST_MED_ZERO_LEFT) / (DIST_MED_PEAK - DIST_MED_ZERO_LEFT)
    else:
        return (DIST_MED_ZERO_RIGHT - d) / (DIST_MED_ZERO_RIGHT - DIST_MED_PEAK)

def dist_high(d):
    if d <= DIST_HIGH_ZERO_LEFT:
        return 0.0
    elif d >= DIST_HIGH_FULL:
        return 1.0
    else:
        return (d - DIST_HIGH_ZERO_LEFT) / (DIST_HIGH_FULL - DIST_HIGH_ZERO_LEFT)

# ------------------------------------------------------------
# Fuzzy Inference
# ------------------------------------------------------------
def fuzzy_inference(oal, dist):
    mu_oal_low  = oal_low(oal)
    mu_oal_med  = oal_med(oal)
    mu_oal_high = oal_high(oal)
    
    mu_dist_low  = dist_low(dist)
    mu_dist_med  = dist_med(dist)
    mu_dist_high = dist_high(dist)
    
    rules = []
    rules.append((min(mu_oal_low, mu_dist_low),  PA_LOW_DIST_LOW))
    rules.append((min(mu_oal_low, mu_dist_med),  PA_LOW_DIST_MED))
    rules.append((min(mu_oal_low, mu_dist_high), PA_LOW_DIST_HIGH))
    rules.append((min(mu_oal_med, mu_dist_low),  PA_MED_DIST_LOW))
    rules.append((min(mu_oal_med, mu_dist_med),  PA_MED_DIST_MED))
    rules.append((min(mu_oal_med, mu_dist_high), PA_MED_DIST_HIGH))
    rules.append((mu_oal_high, PA_OAL_HIGH))
    
    numerator = sum(firing * pa for firing, pa in rules)
    denominator = sum(firing for firing, pa in rules)
    if denominator == 0:
        return 0.0
    return numerator / denominator

# ------------------------------------------------------------
# Data Preprocessing Functions
# ------------------------------------------------------------
def moving_average_filter(series, window=FILTER_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()

def prepare_time_features(df):
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Hour'].astype(str) + ':00:00')
    # Trend: hours elapsed since the first timestamp
    df_copy['t'] = (df_copy['datetime'] - df_copy['datetime'].min()).dt.total_seconds() / 3600
    # Hour of day (cyclic)
    df_copy['hour'] = df_copy['datetime'].dt.hour
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    # Day of week (cyclic)
    df_copy['dow'] = df_copy['datetime'].dt.dayofweek
    df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['dow'] / 7)
    df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['dow'] / 7)
    # Select features for the model
    return df_copy[['t', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']]

# ------------------------------------------------------------
# SVR Prediction Pipeline (Multi-Output)
# ------------------------------------------------------------
def train_svr_model(train_df):
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    X_train = prepare_time_features(train_df)
    # Apply moving average filter to each sensor series
    Y_train = pd.DataFrame({col: moving_average_filter(train_df[col], window=FILTER_WINDOW) for col in sensor_cols})
    
    svr = SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON)
    multi_svr = MultiOutputRegressor(svr)
    # Pipeline with scaling for the features
    model = make_pipeline(StandardScaler(), multi_svr)
    model.fit(X_train, Y_train)
    return model

def make_svr_forecast(model, test_df):
    X_test = prepare_time_features(test_df)
    # Predict returns an array with shape (n_samples, 5) corresponding to sensors P1...P5
    return model.predict(X_test)

# ------------------------------------------------------------
# Main Function: Train and Predict using SVR
# ------------------------------------------------------------
def train_and_predict_svr(train_file='train.csv', test_file='test.csv',
                          output_file='predictions_svr.csv'):
    """
    Steps:
      1) Train multi-output SVR on train.csv for sensors (P1..P5). 
         The sensor series are pre-smoothed using a moving average filter.
      2) Forecast test.csv for sensors, producing PP1..PP5.
      3) Compute:
         - OL2 = L2 norm of observed [P1..P5]
         - PL2 = L2 norm of predicted [PP1..PP5]
         - Dist = Euclidean distance between observed & predicted vectors
         - OAL = OL2/HAL
         - PA = fuzzy_inference(OAL, Dist)
         - RMSE across all sensors
      4) Write everything to output_file.
    """
    print("Script has started running...")
    # Load data
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    # 1) Train SVR model using all sensors simultaneously
    print(f"Training SVR model with FILTER_WINDOW={FILTER_WINDOW}, kernel={SVR_KERNEL}, C={SVR_C}, epsilon={SVR_EPSILON}...")
    model = train_svr_model(train_df)
    
    # 2) Forecast on test data
    forecast_test = make_svr_forecast(model, test_df)
    # Replace any negative predictions with zero
    forecast_test[forecast_test < 0] = 0
    
    # Build output from test_df
    output_df = test_df.copy()
    for i, s in enumerate(sensor_cols):
        output_df[f'PP{s[1]}'] = forecast_test[:, i]
    
    # Compute PL2 (predicted norm) and OL2 (observed norm)
    pred_cols = [f'PP{s[1]}' for s in sensor_cols]
    preds = output_df[pred_cols].values.astype(float)
    obs = output_df[sensor_cols].values.astype(float)
    
    output_df['PL2'] = np.linalg.norm(preds, axis=1)
    output_df['OL2'] = np.linalg.norm(obs, axis=1)
    
    # Dist = Euclidean distance between observed & predicted sensor vectors
    dists = [np.linalg.norm(obs[i] - preds[i]) for i in range(len(output_df))]
    output_df['Dist'] = dists
    
    # OAL = OL2 / HAL (assumes HAL is a column in test_df)
    output_df['OAL'] = output_df['OL2'] / output_df['HAL']
    
    # Fuzzy anomaly probability using OAL & Dist
    print("Computing fuzzy anomaly probability (PA) using OAL & Dist...")
    pa_list = [fuzzy_inference(row['OAL'], row['Dist']) for _, row in output_df.iterrows()]
    output_df['PA'] = pa_list
    
    # RMSE calculation across all sensors
    diff = obs - preds
    rmse = np.sqrt(np.mean(diff**2))
    output_df['RMSE'] = rmse
    
    print("\nEvaluation Metrics:")
    print(f"Mean Dist (Euclidean): {output_df['Dist'].mean():.4f}")
    print(f"Std Dist: {output_df['Dist'].std():.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    anomalies = output_df[output_df['PA'] > 0.7]
    if not anomalies.empty:
        print("\nRows with high anomaly probability (PA>0.7):")
        print(anomalies)
    else:
        print("\nNo anomalies detected (PA>0.7).")
    print(f"Total anomalies: {len(anomalies)}")
    
    # Save the results
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}. Script finished successfully.")

if __name__ == '__main__':
    train_and_predict_svr()
