"""
    Author: Matthew Vallance 001225832
    Purpose: Read data from CSV
    Date: 20/11/23
"""
import csv
import random


def get_data(filename):
    x = []
    y = []
    with open('./example_data/'+filename, 'r') as file:
        # Read CSV file
        data = csv.reader(file)

        # Use a count to ignore headers
        count = 0
        for line in data:
            if count > 1:
                x.append(line[0])
                y.append(line[1])
            count += 1

    return x, y

def create_random_chart_data():
    x = []
    y = []
    # Create for loops for however many 'clusters' we want to generate
    for i in range(1,25):
        x.append(random.randint(0, 25))
        y.append(random.randint(0, 25))

    for i in range(1,25):
        x.append(random.randint(25, 50))
        y.append(random.randint(25, 50))

    for i in range(1,25):
        x.append(random.randint(75, 100))
        y.append(random.randint(75, 100))

    return x, y
