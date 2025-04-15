import csv
from datetime import datetime, timedelta

def process_sensor_data(input_file, output_file):
    # Define which sensor types to keep
    valid_sensor_prefixes = ('M', 'D')
    
    # Dictionary to store the last event time for each sensor
    last_event_times = {}
    
    # Add counters for debugging
    total_lines = 0
    processed_lines = 0
    filtered_by_time = 0

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Date', 'Time', 'Hour', 'Sensor_ID', 'Sensor_event'])

        for line in infile:
            total_lines += 1
            
            fields = line.strip().split(',')
            
            if len(fields) < 7:
                print(f"Skipped line {total_lines}: Insufficient fields")
                continue

            try:
                # Extract timestamp and parse it
                timestamp_str = fields[0] + ' ' + fields[1]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                
                sensor_id = fields[2]
                
                # Check if this is a sensor type we want to keep
                if not any(sensor_id.startswith(prefix) for prefix in valid_sensor_prefixes):
                    continue
                
                event = fields[5]
                
                # Filter out unwanted events based on sensor type
                if (sensor_id.startswith('M') and event == 'OFF') or \
                   (sensor_id.startswith('D') and event == 'CLOSE'):
                    continue
                
                # Check time difference with previous event from the same sensor
                if sensor_id in last_event_times:
                    time_diff = timestamp - last_event_times[sensor_id]
                    if time_diff < timedelta(minutes=1):
                        filtered_by_time += 1
                        if processed_lines % 1000 == 0:
                            print(f"Filtered event from {sensor_id} - too close to previous event: {time_diff.total_seconds():.2f} seconds")
                        continue
                
                # Update the last event time for this sensor
                last_event_times[sensor_id] = timestamp
                
                # Format the output data
                date = timestamp.strftime('%Y-%m-%d')
                time = timestamp.strftime('%H:%M:%S')
                hour = timestamp.strftime('%H')
                
                # Write the filtered data
                writer.writerow([date, time, hour, sensor_id, event])
                processed_lines += 1
                
                if processed_lines % 1000 == 0:
                    print(f"Processed {processed_lines} lines...")
                
            except (ValueError, IndexError) as e:
                print(f"Error processing line {total_lines}: {str(e)}")
                continue

    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total lines read: {total_lines}")
    print(f"Lines written to output: {processed_lines}")
    print(f"Events filtered due to time proximity: {filtered_by_time}")

if __name__ == "__main__":
    input_file = 'raw_data.csv'
    output_file = 'filtered_data.csv'
    
    try:
        process_sensor_data(input_file, output_file)
    except Exception as e:
        print(f"An error occurred: {str(e)}")