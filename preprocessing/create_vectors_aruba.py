from collections import defaultdict
from datetime import datetime, timedelta

def get_sensor_group(sensor_id):
    sensor_groups = {
        'P1': {'M001', 'M002', 'M003', 'M005', 'M006', 'M007', 'M023', 'M024'},
        'P2': {'M004', 'M029', 'M031'},
        'P3': {'M014', 'M015', 'M017', 'M018', 'M019'},
        'P4': {'M009', 'M010', 'M012', 'M013', 'M020', 'M025', 'M026', 'M027', 'M028'},
        'P5': {'D001', 'D002', 'D003', 'D004', 'M008', 'M011', 'M021', 'M022', 'M030'}
    }
    for group, sensors in sensor_groups.items():
        if sensor_id in sensors:
            return group
    return None

def get_date_range(input_file):
    first_date = None
    last_date = None
    with open(input_file, 'r') as f:
        next(f)
        first_line = next(f).strip().split(',')
        first_date = datetime.strptime(first_line[0], '%Y-%m-%d')
        for line in f:
            last_line = line.strip().split(',')
        last_date = datetime.strptime(last_line[0], '%Y-%m-%d')
    return first_date, last_date

def process_vectors(input_file, output_file):
    start_date, end_date = get_date_range(input_file)
    counts = defaultdict(lambda: defaultdict(int))
    with open(input_file, 'r') as f:
        next(f)
        for line in f:
            date, time, hour, sensor_id, event = line.strip().split(',')
            group = get_sensor_group(sensor_id)
            if group:
                counts[(date, hour)][group] += 1

    with open(output_file, 'w') as f:
        f.write('Date,Hour,P1,P2,P3,P4,P5\n')
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            for hour in range(24):
                hour_str = f"{hour:02d}"

                p1 = min(counts[(date_str, hour_str)]['P1'], 40)
                p2 = min(counts[(date_str, hour_str)]['P2'], 40)
                p3 = min(counts[(date_str, hour_str)]['P3'], 40)
                p4 = min(counts[(date_str, hour_str)]['P4'], 40)
                p5 = min(counts[(date_str, hour_str)]['P5'], 40)

                f.write(f'{date_str},{hour_str},{p1},{p2},{p3},{p4},{p5}\n')

            current_date += timedelta(days=1)

def main():
    input_file = 'filtered_data_aruba.csv'
    output_file = 'vectors_aruba.csv'
    try:
        process_vectors(input_file, output_file)
        print(f"Successfully created {output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
