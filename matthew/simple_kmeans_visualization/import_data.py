"""
    Author: Matthew Vallance 001225832
    Purpose: Read data from CSV
    Date: 20/11/23
"""
import csv
import random
import testing_functions


def get_data(filename):
    x = []
    y = []
    with open('example_data/'+filename, 'r') as file:
        # Read CSV file
        data = csv.reader(file)

        # Use a count to ignore headers
        count = 0
        for line in data:
            if count > 1:
                x.append(int(line[0]))
                y.append(int(line[1]))
            count += 1

    return x, y

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
    x, y = create_random_chart_data()
    data = list(zip(x, y))
    testing_functions.write_to_csv(data, 'lrg_rand_clusters')