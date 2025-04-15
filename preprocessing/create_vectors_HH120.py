import csv
from datetime import datetime, timedelta
from collections import defaultdict

def get_sensor_groups():
    return {
        'P1': {'M009', 'M010', 'M011', 'MA016'},
        'P2': {'M008', 'MA018'},
        'P3': {'MA017'},
        'P4': {'M003', 'M004', 'M005', 'M006', 'M007', 'MA015'},
        'P5': {'D002', 'D004', 'M001', 'M002', 'M011'}
    }

def create_hourly_vectors(input_file, output_file):
    daily_counts = defaultdict(lambda: {str(h).zfill(2): {f'P{i}': 0 for i in range(1, 6)} 
                                        for h in range(24)})

    sensor_groups = get_sensor_groups()

    min_date = None
    max_date = None

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)

        for row in reader:
            date, _, hour, sensor_id, _ = row
            current_date = datetime.strptime(date, '%Y-%m-%d').date()
            
            if min_date is None or current_date < min_date:
                min_date = current_date
            if max_date is None or current_date > max_date:
                max_date = current_date
            
            for group_name, sensors in sensor_groups.items():
                if sensor_id in sensors:
                    daily_counts[date][hour][group_name] += 1
    
    # Limitarea valorilor la maxim 60
    for date in daily_counts:
        for hour in daily_counts[date]:
            for group in daily_counts[date][hour]:
                if daily_counts[date][hour][group] > 35:
                    daily_counts[date][hour][group] = 35

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Date', 'Hour', 'P1', 'P2', 'P3', 'P4', 'P5'])
        
        current_date = min_date
        while current_date <= max_date:
            date_str = current_date.strftime('%Y-%m-%d')
            for hour in range(24):
                hour_str = str(hour).zfill(2)
                row = [
                    date_str,
                    hour_str,
                    daily_counts[date_str][hour_str]['P1'],
                    daily_counts[date_str][hour_str]['P2'],
                    daily_counts[date_str][hour_str]['P3'],
                    daily_counts[date_str][hour_str]['P4'],
                    daily_counts[date_str][hour_str]['P5']
                ]
                writer.writerow(row)
            
            current_date += timedelta(days=1)
    
    return min_date, max_date

if __name__ == "__main__":
    input_file = 'filtered_data_HH120.csv'
    output_file = 'vectors_HH120.csv'
    
    try:
        min_date, max_date = create_hourly_vectors(input_file, output_file)
        print("Processing complete!")
        print(f"Date range covered: {min_date} to {max_date}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
