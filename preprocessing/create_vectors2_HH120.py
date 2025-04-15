import pandas as pd
import numpy as np

def main():
    # Load the input file "vectors_HH120.csv"
    df = pd.read_csv("vectors_HH120.csv")
    
    # Compute the L2 norm of the sensor vector [P1, P2, P3, P4, P5] for each row
    # This is computed as sqrt(P1^2 + P2^2 + P3^2 + P4^2 + P5^2)
    sensor_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
    df['L2'] = np.sqrt((df[sensor_cols]**2).sum(axis=1))
    
    # Compute HAL (Hourly Activity Level): for each Hour (0-23),
    # calculate the average L2 norm across all days.
    hal_df = df.groupby("Hour")["L2"].mean().reset_index().rename(columns={"L2": "HAL"})
    
    # Merge the HAL back into the original dataframe on the Hour column.
    df = pd.merge(df, hal_df, on="Hour", how="left")
    
    # Save the output to "vectors2_HH120.csv"
    df.to_csv("vectors2_HH120.csv", index=False)
    print("Output saved to vectors2_HH120.csv")

if __name__ == '__main__':
    main()
