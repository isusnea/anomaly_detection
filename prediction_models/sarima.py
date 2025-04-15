import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================
#                   USER TUNING SECTION
#    Modify these values below to tune the membership functions
#    for OAL (Observed Activity Level) and Dist (Euclidean),
#    the moving average filter, and now the SARIMA model.
# ============================================================

FILTER_WINDOW = 3  # <-- Adjust the moving average filter window here

# ---- SARIMA TUNING PARAMETERS ----
# These parameters are used for each sensor series.
SARIMA_ORDER = (2, 0, 1)                # Non-seasonal (p,d,q)
SARIMA_SEASONAL_ORDER = (2, 0, 1, 24)     # Seasonal (P,D,Q,s): s=24 for daily seasonality

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

def prepare_sarima_series(df, sensor):
    # Apply moving average filter and set datetime index
    series = moving_average_filter(df[sensor], window=FILTER_WINDOW)
    dt = pd.to_datetime(df['Date'] + ' ' + df['Hour'].astype(str) + ':00:00')
    dt = dt.sort_values()  # ensure the timestamps are in order
    series.index = dt
    # Explicitly set frequency to hourly using 'h' to remove deprecation warning
    series = series.asfreq('h')
    return series

# ------------------------------------------------------------
# SARIMA Prediction Pipeline (Univariate for each sensor)
# ------------------------------------------------------------
def train_sarima_models(train_df):
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    models = {}
    for sensor in sensor_cols:
        series = prepare_sarima_series(train_df, sensor)
        model = SARIMAX(series,
                        order=SARIMA_ORDER,
                        seasonal_order=SARIMA_SEASONAL_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        models[sensor] = fitted_model
    return models

def make_sarima_forecasts(models, test_df):
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    forecasts = {}
    for sensor in sensor_cols:
        # Forecast for the number of periods equal to test_df's length
        forecast = models[sensor].forecast(steps=len(test_df))
        # Replace negative forecasts with zero
        forecast[forecast < 0] = 0
        forecasts[sensor] = forecast.values
    return forecasts

# ------------------------------------------------------------
# Main Function: Train and Predict using SARIMA
# ------------------------------------------------------------
def train_and_predict_sarima(train_file='train.csv', test_file='test.csv',
                             output_file='predictions_sarima.csv'):
    """
    Steps:
      1) Train SARIMA models on train.csv for each sensor (P1..P5).
         The sensor series are pre-smoothed using a moving average filter.
      2) Forecast test.csv for each sensor, producing PP1..PP5.
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
    
    # 1) Train SARIMA models for each sensor
    print(f"Training SARIMA models with FILTER_WINDOW={FILTER_WINDOW}, SARIMA_ORDER={SARIMA_ORDER}, "
          f"SARIMA_SEASONAL_ORDER={SARIMA_SEASONAL_ORDER}...")
    models = train_sarima_models(train_df)
    
    # 2) Forecast on test data for each sensor
    forecasts = make_sarima_forecasts(models, test_df)
    
    # Build output from test_df
    output_df = test_df.copy()
    for sensor in sensor_cols:
        # Create predicted column names like PP1, PP2, etc.
        output_df[f'PP{sensor[1]}'] = forecasts[sensor]
    
    # Compute PL2 (predicted norm) and OL2 (observed norm)
    pred_cols = [f'PP{s[1]}' for s in sensor_cols]
    preds = output_df[pred_cols].values.astype(float)
    obs = output_df[sensor_cols].values.astype(float)
    
    output_df['PL2'] = np.linalg.norm(preds, axis=1)
    output_df['OL2'] = np.linalg.norm(obs, axis=1)
    
    # Euclidean distance between observed & predicted sensor vectors
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
    train_and_predict_sarima()
