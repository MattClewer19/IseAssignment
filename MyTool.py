import pandas as pd
import numpy as np
import os
import itertools
import random

def create_interval(bounds, k):
    # Takes the upper and lower bounds of a config option and returns the different subregions of this option

    low, high = bounds

    step = (high - low) / k
    param_intervals = []
    
    for i in range(k):
        start = low + i * step
        end = low + (i + 1) * step
        param_intervals.append((start, end))
    
    return param_intervals

def sample_configuration(intervals, data_columns):
    # For each parameter, select one of the intervals, select a random value in that interval, append these to create a configuration
    config = []

    for param_intervals, col_values in zip(intervals, data_columns):
        interval = random.choice(param_intervals)

        # Binary case
        if interval[0] == interval[1]:
            value = interval[0]
        else:
            sampled_value = random.uniform(interval[0], interval[1])

            # Find closest existing value in the data for this parameter
            value = min(col_values, key=lambda x: abs(x - sampled_value))

        config.append(value)

    return config

def explorationPhase(data, n, k):
    #For each column, generate the appropriate intervals
    intervals = []

    for col_name in data.columns[:-1]:
        column = data[col_name]

        unique_vals = np.unique(column)
        if np.all(np.isin(unique_vals, [0, 1])):
            intervals.append([(0, 0), (1, 1)])
        else:
            bounds = (column.min(), column.max())
            intervals.append(create_interval(bounds, k))

    #Sample from n different subspaces
    data_columns = [data[col].values for col in data.columns[:-1]]
    samples = []
    for _ in range(n):
        samples.append(sample_configuration(intervals, data_columns))

    return samples
        


def get_performance(sample, data, configColumns, performanceColumn):
    matched_row = data.loc[
        (data[configColumns] == pd.Series(sample, index=configColumns)).all(axis=1)
    ]

    if not matched_row.empty:
        return matched_row[performanceColumn].iloc[0]
    else:
        return None  # or worst_value



def mySearch(filePath, budget, outputFile, buget):
    # read data into matrix
    data = pd.read_csv(filePath)

    # seperate feature columns and label
    configColumns = data.columns[:-1]
    performanceColumn = data.columns[-1]

    # # decide if its a maximisation problem
    if os.path.basename(filePath).split('.')[0] == "---":
        maximisation = True
    else:
        maximisation = False

    # # Extract the best and worst performance values
    if maximisation:
        worst_value = -np.inf
    else:
        worst_value = np.inf

    # Initialize the best solution and performance
    # best_performance = -np.inf if maximisation else np.inf
    # best_solution = []

    # Store all search results
    # search_results = []

    samples = explorationPhase(data, int(0.3*budget), 3)
    performances = []
    for sample in samples:
        performance = get_performance(sample, data, configColumns, performanceColumn)
        if performance != None:
            # Existing configuration
            performances.append(performance)
        # else:
        #     # Non-existing configuration
        #     print("Non-existing configuration:", sample)
        
    print(performances)
    print(len(performances))

    

def main():
    dataPath = "data"
    resultsPath = "results"

    results = {}
    budget = 100

    for file_name in os.listdir(dataPath):
        if file_name.endswith(".csv"):
            filePath = os.path.join(dataPath, file_name)
            outputFile = os.path.join(resultsPath, file_name)
            mySearch(filePath, budget, outputFile, budget)
            # best_solution, best_performance = mySearch(filePath, budget, outputFile)
            # results[file_name] = {
            #     "Best Solution": best_solution,
            #     "Best Performance": best_performance
            # }

    # Print the results
    for system, result in results.items():
        print(f"System: {system}")
        print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
        print(f"  Best Performance: {result['Best Performance']}")

if __name__ == "__main__":
    main()