import sys
import csv

def split_file(num_train_days, input_file="vectors2.csv", train_file="train.csv", test_file="test.csv"):
    """
    Splits the input CSV file into train and test files.
    
    - The input file has the header:
      Date,Time,Sensor_ID,Loc1,Loc2,Event,Type,Activity
    - The first column (Date) is in the format yyyy-mm-dd.
    - The script groups rows by Date (assuming the file is sorted by date).
    - The first `num_train_days` unique days are written to train_file.
    - The remaining rows are written to test_file.
    """
    with open(input_file, "r", newline="") as infile, \
         open(train_file, "w", newline="") as train_out, \
         open(test_file, "w", newline="") as test_out:
        
        reader = csv.reader(infile)
        train_writer = csv.writer(train_out)
        test_writer = csv.writer(test_out)
        
        # Read and write the header to both output files.
        header = next(reader)
        train_writer.writerow(header)
        test_writer.writerow(header)
        
        current_date = None
        train_days_count = 0
        
        for row in reader:
            if len(row) < 2:
                continue  # Skip rows without at least Date and Time.
            
            row_date = row[0].strip()
            
            # Whenever the date changes, increment the count of unique days.
            if row_date != current_date:
                current_date = row_date
                train_days_count += 1
            
            # Write to train.csv if we are still within the train days; otherwise, to test.csv.
            if train_days_count <= num_train_days:
                train_writer.writerow(row)
            else:
                test_writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split.py <number_of_train_days>")
        sys.exit(1)
    
    try:
        num_days = int(sys.argv[1])
    except ValueError:
        print("The number of days must be an integer.")
        sys.exit(1)
    
    split_file(num_days)
