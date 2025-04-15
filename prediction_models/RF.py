import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ============================================================
#                   USER TUNING SECTION
#    Modify these values below to tune the membership functions
#    for OAL (Observed Activity Level) and Dist (Euclidean),
#    the moving average filter, and now the Random Forest model.
# ============================================================

FILTER_WINDOW = 3  # <-- Adjust the moving average filter window here

# ---- Random Forest Tuning Parameters ----
RF_N_ESTIMATORS = 100      # Number of trees in the forest
RF_MAX_DEPTH = 5        # Maximum depth of the tree (None means unlimited)
RF_RANDOM_STATE = 42       # Random state for reproducibility

# ---- OAL THRESHOLDS ----
OAL_LOW_MAX = 0.0      # Full membership if OAL <= this
OAL_LOW_ZERO = 0.2     # Zero membership if OAL >= this (linear in between)

OAL_MED_ZERO_LEFT = 0.1   # OAL <= this => no membership for MED
OAL_MED_PEAK = 1          # Peak membership for MED
OAL_MED_ZERO_RIGHT = 1.7  # OAL >= this => no membership for MED

OAL_HIGH_ZERO_LEFT = 1.2  # OAL <= this => membership 0
OAL_HIGH_FULL = 2.5        # OAL >= this => membership 1.0 (linear in between)

# ---- DIST THRESHOLDS (Euclidean Distance) ----
DIST_LOW_MAX = 5.0     # Full membership if Dist <= this
DIST_LOW_ZERO = 12.0    # Zero membership if Dist >= this (linear in between)

DIST_MED_ZERO_LEFT = 6.0  # Dist <= this => no membership for MED
DIST_MED_PEAK = 10.0       # Dist at which MED is max
DIST_MED_ZERO_RIGHT = 14.0 # Dist >= this => no membership for MED

DIST_HIGH_ZERO_LEFT = 10.0 # Dist <= this => membership 0
DIST_HIGH_FULL = 15.0      # Dist >= this => membership 1.0 (linear in between)

# Singletons for rule outputs
PA_LOW_DIST_LOW  = 0.65
PA_LOW_DIST_MED  = 0.8
PA_LOW_DIST_HIGH = 0.9

PA_MED_DIST_LOW  = 0.3
PA_MED_DIST_MED  = 0.5
PA_MED_DIST_HIGH = 0.7


PA_OAL_HIGH      = 0.1

# ============================================================
#                END OF USER TUNING SECTION
# ============================================================

# ------------------------------------------------------------
# Fuzzy Membership Functions for OAL and Dist
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
    """
    Creates a feature DataFrame from the Date and Hour columns.
    Features include a time trend and cyclical representations for hour and day-of-week.
    """
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Hour'].astype(str) + ':00:00')
    df_copy['t'] = (df_copy['datetime'] - df_copy['datetime'].min()).dt.total_seconds() / 3600
    df_copy['hour'] = df_copy['datetime'].dt.hour
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    df_copy['dow'] = df_copy['datetime'].dt.dayofweek
    df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['dow'] / 7)
    df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['dow'] / 7)
    return df_copy[['t', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']]

# ------------------------------------------------------------
# Random Forest Prediction Pipeline (Multivariate)
# ------------------------------------------------------------
def train_rf_model(train_df):
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    X_train = prepare_time_features(train_df)
    # Apply moving average filter to smooth each sensor series
    Y_train = pd.DataFrame({col: moving_average_filter(train_df[col], window=FILTER_WINDOW)
                            for col in sensor_cols})
    
    model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                  max_depth=RF_MAX_DEPTH,
                                  random_state=RF_RANDOM_STATE)
    model.fit(X_train, Y_train)
    return model

def make_rf_forecast(model, test_df):
    X_test = prepare_time_features(test_df)
    forecast = model.predict(X_test)
    # Replace any negative predictions with zero
    forecast[forecast < 0] = 0
    return forecast

# ------------------------------------------------------------
# Main Function: Train and Predict using Random Forest Regression
# ------------------------------------------------------------
def train_and_predict_rf(train_file='train.csv', test_file='test.csv',
                         output_file='predictions_rf.csv'):
    """
    Steps:
      1) Train a Random Forest model on train.csv for sensors (P1..P5).
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
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    # 1) Train Random Forest model
    print(f"Training Random Forest with RF_N_ESTIMATORS={RF_N_ESTIMATORS}, RF_MAX_DEPTH={RF_MAX_DEPTH}...")
    model = train_rf_model(train_df)
    
    # 2) Forecast test data
    forecast = make_rf_forecast(model, test_df)
    
    # Build output DataFrame from test_df
    output_df = test_df.copy()
    for i, sensor in enumerate(sensor_cols):
        # Create predicted column names like PP1, PP2, etc.
        output_df[f'PP{sensor[1]}'] = forecast[:, i]
    
    # Compute norms and distances
    pred_cols = [f'PP{s[1]}' for s in sensor_cols]
    preds = output_df[pred_cols].values.astype(float)
    obs = output_df[sensor_cols].values.astype(float)
    
    output_df['PL2'] = np.linalg.norm(preds, axis=1)
    output_df['OL2'] = np.linalg.norm(obs, axis=1)
    dists = [np.linalg.norm(obs[i] - preds[i]) for i in range(len(output_df))]
    output_df['Dist'] = dists
    output_df['OAL'] = output_df['OL2'] / output_df['HAL']
    
    print("Computing fuzzy anomaly probability (PA) using OAL & Dist...")
    pa_list = [fuzzy_inference(row['OAL'], row['Dist']) for _, row in output_df.iterrows()]
    output_df['PA'] = pa_list
    
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
    
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}. Script finished successfully.")

if __name__ == '__main__':
    train_and_predict_rf()
