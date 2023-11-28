"""
    Author: Matthew Vallance 001225832
    Purpose: Simple script to put data into a CSV for comparison externally
    Date: 26/10/23
"""
import csv

def write_to_csv(data, filename='testing_data'):
    # Routine to output any data into CSVs
    with open(filename+".csv", "w") as file_to_write:
        # Create writer to write row
        writer = csv.writer(file_to_write)
        # Write each row of data to the CSV file
        for row in data:
            writer.writerow(row)
        print('Generated ' + filename + '.csv')