"""
    Author: Matthew Vallance 001225832
    Purpose: Read data from CSV
    Date: 20/11/23
"""
import csv
import random
import pandas as pd


def get_dataframe(filename):
    with open('inputs/' + filename, 'r') as file:
        # Read CSV file
        data = csv.reader(file)

        # Create a dataframe
        count = 0
        for line in data:
            temp_line_data = []
            if count < 1:
                df = pd.DataFrame(columns=line)
                row_count = len(line)
            else:
                for col in range(0, row_count):
                    temp_line_data.append(int(line[col]))
                df.loc[count] = temp_line_data

            count += 1

    return df


# Legacy code for testing
def create_random_chart_data():
    x = []
    y = []
    # Create for loops for however many 'clusters' we want to generate
    for i in range(1, 1000):
        x.append(random.randint(0, 250))
        y.append(random.randint(0, 250))

    for i in range(1, 1000):
        x.append(random.randint(0, 250))
        y.append(random.randint(750, 1000))

    for i in range(1, 1000):
        x.append(random.randint(750, 1000))
        y.append(random.randint(750, 1000))

    # Chuck in some random points
    for i in range(1, 3000):
        x.append(random.randint(0, 1000))
        y.append(random.randint(0, 1000))

    return x, y


if __name__ == "__main__":
    print(get_dataframe('segmentation_data.csv').head())
    exit()
