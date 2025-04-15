import pandas as pd
import numpy as np
from prophet import Prophet

# ============================================================
#                   USER TUNING SECTION
#    Modify these values below to tune the membership functions
#    for OAL (Observed Activity Level) and Dist (Euclidean).
# ============================================================

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
# Fuzzy Membership Functions for OAL
# ------------------------------------------------------------
def oal_low(x):
    """
    Full membership if x <= OAL_LOW_MAX, zero if x >= OAL_LOW_ZERO, linear in between.
    """
    if x <= OAL_LOW_MAX:
        return 1.0
    elif x >= OAL_LOW_ZERO:
        return 0.0
    else:
        return 1.0 - (x - OAL_LOW_MAX) / (OAL_LOW_ZERO - OAL_LOW_MAX)

def oal_med(x):
    """
    Zero membership if x <= OAL_MED_ZERO_LEFT or x >= OAL_MED_ZERO_RIGHT.
    Peak membership around OAL_MED_PEAK.
    """
    if x <= OAL_MED_ZERO_LEFT or x >= OAL_MED_ZERO_RIGHT:
        return 0.0
    elif x < OAL_MED_PEAK:
        return (x - OAL_MED_ZERO_LEFT) / (OAL_MED_PEAK - OAL_MED_ZERO_LEFT)
    else:
        return (OAL_MED_ZERO_RIGHT - x) / (OAL_MED_ZERO_RIGHT - OAL_MED_PEAK)

def oal_high(x):
    """
    Zero membership if x <= OAL_HIGH_ZERO_LEFT, full membership if x >= OAL_HIGH_FULL,
    linear in between.
    """
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
    """
    Full membership if d <= DIST_LOW_MAX, zero if d >= DIST_LOW_ZERO, linear in between.
    """
    if d <= DIST_LOW_MAX:
        return 1.0
    elif d >= DIST_LOW_ZERO:
        return 0.0
    else:
        return (DIST_LOW_ZERO - d) / (DIST_LOW_ZERO - DIST_LOW_MAX)

def dist_med(d):
    """
    Zero membership if d <= DIST_MED_ZERO_LEFT or d >= DIST_MED_ZERO_RIGHT.
    Peak membership around DIST_MED_PEAK.
    """
    if d <= DIST_MED_ZERO_LEFT or d >= DIST_MED_ZERO_RIGHT:
        return 0.0
    elif d < DIST_MED_PEAK:
        return (d - DIST_MED_ZERO_LEFT) / (DIST_MED_PEAK - DIST_MED_ZERO_LEFT)
    else:
        return (DIST_MED_ZERO_RIGHT - d) / (DIST_MED_ZERO_RIGHT - DIST_MED_PEAK)

def dist_high(d):
    """
    Zero membership if d <= DIST_HIGH_ZERO_LEFT, full membership if d >= DIST_HIGH_FULL,
    linear in between.
    """
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
    """
    Compute anomaly probability PA using fuzzy rules:
      - OAL=LOW, Dist=LOW   => PA_LOW_DIST_LOW
      - OAL=LOW, Dist=MED   => PA_LOW_DIST_MED
      - OAL=LOW, Dist=HIGH  => PA_LOW_DIST_HIGH
      - OAL=MED, Dist=LOW   => PA_MED_DIST_LOW
      - OAL=MED, Dist=MED   => PA_MED_DIST_MED
      - OAL=MED, Dist=HIGH  => PA_MED_DIST_HIGH
      - OAL=HIGH => PA_OAL_HIGH (for all Dist)
    """
    mu_oal_low  = oal_low(oal)
    mu_oal_med  = oal_med(oal)
    mu_oal_high = oal_high(oal)
    
    mu_dist_low  = dist_low(dist)
    mu_dist_med  = dist_med(dist)
    mu_dist_high = dist_high(dist)
    
    # Build rule base
    rules = []
    # OAL LOW
    rules.append((min(mu_oal_low, mu_dist_low),  PA_LOW_DIST_LOW))
    rules.append((min(mu_oal_low, mu_dist_med),  PA_LOW_DIST_MED))
    rules.append((min(mu_oal_low, mu_dist_high), PA_LOW_DIST_HIGH))
    # OAL MED
    rules.append((min(mu_oal_med, mu_dist_low),  PA_MED_DIST_LOW))
    rules.append((min(mu_oal_med, mu_dist_med),  PA_MED_DIST_MED))
    rules.append((min(mu_oal_med, mu_dist_high), PA_MED_DIST_HIGH))
    # OAL HIGH => PA_OAL_HIGH
    rules.append((mu_oal_high, PA_OAL_HIGH))
    
    numerator = sum(firing * pa for firing, pa in rules)
    denominator = sum(firing for firing, pa in rules)
    if denominator == 0:
        return 0.0
    return numerator / denominator

# ------------------------------------------------------------
# Prophet Prediction Pipeline
# ------------------------------------------------------------
def moving_average_filter(series, window=3):
    return series.rolling(window=window, min_periods=1, center=True).mean()

def prepare_prophet_df(df, series_col, filter_window=None):
    df_prophet = df.copy()
    if filter_window is not None:
        df_prophet[series_col] = moving_average_filter(df_prophet[series_col], window=filter_window)
    df_prophet['ds'] = pd.to_datetime(df_prophet['Date'] + ' ' + df_prophet['Hour'].astype(str) + ':00:00')
    df_prophet = df_prophet[['ds', series_col]].rename(columns={series_col: 'y'})
    return df_prophet

def train_prophet_model(train_df, series_col, filter_window=None):
    df_prophet = prepare_prophet_df(train_df, series_col, filter_window=filter_window)
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)
    return model

def make_prophet_forecast(model, df):
    future = pd.DataFrame()
    future['ds'] = pd.to_datetime(df['Date'] + ' ' + df['Hour'].astype(str) + ':00:00')
    forecast = model.predict(future)
    return forecast['yhat'].values

def train_and_predict_prophet(train_file='train.csv', test_file='test.csv',
                              output_file='prophet_predictions2.csv', filter_window=3):
    """
    Steps:
      1) Train Prophet on train.csv for each sensor (P1..P5). Negative predictions -> zero.
      2) Forecast test.csv, produce PP1..PP5
      3) Compute:
         - OL2 = L2 norm of observed [P1..P5]
         - PL2 = L2 norm of predicted [PP1..PP5]
         - Dist = Euclidean distance between observed & predicted vectors
         - OAL = OL2/HAL
         - PA = fuzzy_inference(OAL, Dist)
         - RMSE across all sensors
      4) Write everything to output_file
    """
    print("Script has started running...")
    # Load data
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)
    series_names = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    # 1) Train Prophet for each series & produce test predictions
    test_predictions = {}
    for s in series_names:
        print(f"Training Prophet model for {s} with filter_window={filter_window} ...")
        model = train_prophet_model(train_df, s, filter_window=filter_window)
        forecast_test = make_prophet_forecast(model, test_df)
        forecast_test[forecast_test < 0] = 0
        test_predictions[s] = forecast_test
    
    # Build output from test_df
    output_df = test_df.copy()
    for s in series_names:
        output_df[f'PP{s[1]}'] = test_predictions[s]
    
    # Compute PL2, OL2
    pred_cols = [f'PP{s[1]}' for s in series_names]
    preds = output_df[pred_cols].values.astype(float)
    obs = output_df[series_names].values.astype(float)
    
    output_df['PL2'] = np.linalg.norm(preds, axis=1)
    output_df['OL2'] = np.linalg.norm(obs, axis=1)
    
    # Dist = Euclidean distance between observed & predicted sensor vectors
    dists = []
    for i in range(len(output_df)):
        diff_vec = obs[i] - preds[i]
        d_eucl = np.linalg.norm(diff_vec)
        dists.append(d_eucl)
    output_df['Dist'] = dists
    
    # OAL = OL2 / HAL
    output_df['OAL'] = output_df['OL2'] / output_df['HAL']
    
    # Fuzzy anomaly probability
    print("Computing fuzzy anomaly probability (PA) using OAL & Dist...")
    pa_list = []
    for i, row in output_df.iterrows():
        pa_val = fuzzy_inference(row['OAL'], row['Dist'])
        pa_list.append(pa_val)
    output_df['PA'] = pa_list
    
    # RMSE
    diff = obs - preds
    rmse = np.sqrt(np.mean(diff**2))
    output_df['RMSE'] = rmse
    
    print("\nEvaluation Metrics:")
    print(f"Mean Dist (Euclidean): {output_df['Dist'].mean():.4f}")
    print(f"Std Dist: {output_df['Dist'].std():.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Example: show rows with PA > 0.7
    anomalies = output_df[output_df['PA'] > 0.7]
    if not anomalies.empty:
        print("\nRows with high anomaly probability (PA>0.7):")
        print(anomalies)
    else:
        print("\nNo anomalies detected (PA>0.7).")
    print(f"Total anomalies: {len(anomalies)}")
    
    # Save
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}. Script finished successfully.")

if __name__ == '__main__':
    train_and_predict_prophet(filter_window=1)
