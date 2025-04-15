import csv

def process_sensor_data(input_filename='raw_data.csv', output_filename='filtered_data.csv'):
    """
    Processes sensor data from a raw CSV file and writes filtered output.
    
    Filtering rules:
    - Only include sensors of type Mxxx (motion) and Dxxx (door).
    - Exclude temperature sensors (Txxx).
    - For Mxxx sensors, only keep events with value 'ON' (skip 'OFF').
    - For Dxxx sensors, only keep events with value 'OPEN' (skip 'CLOSE').
    - Remove the optional description field.
    
    The output CSV will contain the following columns:
      Date (yyyy-mm-dd),
      Time (hh:mm:ss without decimals),
      Hour (extracted from Time),
      Sensor_ID,
      Sensor_event.
    """
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header row (optional)
        writer.writerow(["Date", "Time", "Hour", "Sensor_ID", "Sensor_event"])
        
        for line in infile:
            # Split the line into tokens; note that the description field (if present) is ignored.
            tokens = line.strip().split()
            # A valid line must have at least 4 tokens: date, time, sensor_id, sensor_event.
            if len(tokens) < 4:
                continue  # Skip malformed lines
            
            date = tokens[0]
            time_full = tokens[1]
            # Remove decimals from seconds (e.g., "00:03:50.209589" -> "00:03:50")
            time_no_decimals = time_full.split('.')[0]
            # Extract hour (first two characters of time_no_decimals)
            hour = time_no_decimals.split(':')[0]
            sensor_id = tokens[2]
            sensor_event = tokens[3]
            
            # Apply filtering based on sensor type and event:
            if sensor_id.startswith('T'):
                continue  # Skip temperature sensors
            if sensor_id.startswith('M') and sensor_event == 'OFF':
                continue  # Skip motion sensor events that are OFF
            if sensor_id.startswith('D') and sensor_event == 'CLOSE':
                continue  # Skip door sensor events that are CLOSE
            
            writer.writerow([date, time_no_decimals, hour, sensor_id, sensor_event])

if __name__ == '__main__':
    process_sensor_data()
