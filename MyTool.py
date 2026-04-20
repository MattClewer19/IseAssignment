import pandas as pd
import numpy as np
import os
import random
from sklearn.tree import DecisionTreeRegressor
import time

# Takes the upper and lower bounds of a config option and returns k different subregions of this option
def CreateInterval(bounds, k):
    low, high = bounds

    step = (high - low) / k
    param_intervals = []
    
    for i in range(k):
        start = low + i * step
        end = low + (i + 1) * step
        param_intervals.append((start, end))
    
    return param_intervals

# For each parameter, select one of the intervals, select a random value in that interval, append these to create a configuration
def SampleConfiguration(intervals, data_columns):
    config = []

    for param_intervals, col_values in zip(intervals, data_columns):
        # Randomly select one of the intervals for this parameter
        interval = random.choice(param_intervals)

        if interval[0] == interval[1]:
            # Binary case
            value = interval[0]
        else:
            # Non-Binary case - sample a value from the selected interval
            sampled_value = random.uniform(interval[0], interval[1])

            # Find closest existing value in the data for this parameter
            value = min(col_values, key=lambda x: abs(x - sampled_value))

        config.append(value)

    return config

# The exploration phase of the solution - sample from n different subspaces, where the subspaces are defined by the intervals created in CreateInterval
def ExplorationPhase(data, k, n):
    intervals = []

    #For each column, generate the appropriate intervals
    for col_name in data.columns[:-1]:
        column = data[col_name]

        unique_vals = np.unique(column)
        if np.all(np.isin(unique_vals, [0, 1])):
            intervals.append([(0, 0), (1, 1)])
        else:
            bounds = (column.min(), column.max())
            intervals.append(CreateInterval(bounds, k))

    #Sample from n different subspaces
    data_columns = [data[col].values for col in data.columns[:-1]]
    samples = []
    while len(samples) < n:
        configuration = SampleConfiguration(intervals, data_columns)
        # Measure performance of this configuratin, then append the configuration and its performance to the list of samples
        sample = MeasurePerformance(configuration, data, data.columns[:-1], data.columns[-1])
        if sample != None:
            samples.append(configuration + [sample])

    return samples

# Simulates measuring the performance of a configuration by finding it in the dataset
def MeasurePerformance(sample, data, configColumns, performanceColumn):
    matched_row = data.loc[(data[configColumns] == pd.Series(sample, index=configColumns)).all(axis=1)]

    if not matched_row.empty:
        return matched_row[performanceColumn].iloc[0]
    else:
        return None
    
# Samples a configuration, normally distributesed around the best solution found so far, then snaps this to the nearest real configuration in the dataset
def SampleNearBest(data, best_solution, scale=0.1):
    new_config = []

    for val, col_name in zip(best_solution, data.columns[:-1]):
        col_values = data[col_name]

        col_min = col_values.min()
        col_max = col_values.max()

        # Set standard deviation relative to range
        sigma = (col_max - col_min) * scale

        # Sample from normal distribution
        sampled = np.random.normal(loc=val, scale=sigma)

        # # Clip to valid bounds
        # sampled = np.clip(sampled, col_min, col_max)

        # Snap to nearest real value in dataset
        snapped = col_values[np.abs(col_values - sampled).argmin()]
        # snapped = min(col_values, key=lambda x: abs(x - sampled))

        new_config.append(snapped)

    return new_config
    
# Given a model, returns a configuration to measure the performance of next
def AcquisitionFunction(data, model, num_samples, best_solution, maximisation):
    samples = []

    # 70% of samples are randomly sampled from the entire configuration space
    for _ in range(int(0.7*num_samples)):
        sampled_config = [int(np.random.choice(data[col].unique())) for col in data.columns[:-1]]
        samples.append(sampled_config + [model.predict([sampled_config])[0]])

    # 30% are sampled from a normal distribution around the best solution found so far
    for _ in range(int(0.3*num_samples)):
        sampled_config = SampleNearBest(data, best_solution, scale=0.1)
        samples.append(sampled_config + [model.predict([sampled_config])[0]])

    if maximisation:
        best_sample = max(samples, key=lambda x: x[-1])
    else:        
        best_sample = min(samples, key=lambda x: x[-1])

    # Returns the best configuration found according to the model
    return best_sample[:-1]

# The exploitation phase of the solution - fits a model to the configurations sampled so far, uses an acquisition function to select a new configuration to sample, measures the performance of this configuration, then adds it to the list of samples
def ExploitationPhase(data, samples, maximisation, n):
    initialLength = len(samples)

    # Does not exceed compute budget
    while len(samples) < initialLength + n:
        if maximisation:
            best_solution = max(samples, key=lambda x: x[-1])
        else:
            best_solution = min(samples, key=lambda x: x[-1])

        X = [s[:-1] for s in samples]
        y = [s[-1] for s in samples]

        # Fits a CART decision tree regression model to the samples collected so far
        model = DecisionTreeRegressor()
        model.fit(X, y)
        
        # Selects configuration to measure next
        newSample = AcquisitionFunction(data, model, 100, best_solution, maximisation)

        # Measures the performance of this configuration, then appends the configuration and its performance to the list of samples
        performance = MeasurePerformance(newSample, data, data.columns[:-1], data.columns[-1])
        if performance != None:
            samples.append(newSample + [performance])

    if maximisation:
        best_solution = max(samples, key=lambda x: x[-1])
    else:
        best_solution = min(samples, key=lambda x: x[-1])

    return best_solution[:-1], best_solution[-1]



def MySearch(filePath, budget):
    # read data into matrix
    data = pd.read_csv(filePath)

    samples = ExplorationPhase(data, 3, int(0.3*budget))
 
    # # decide if its a maximisation problem
    if os.path.basename(filePath).split('.')[0] == "---":
        maximisation = True
    else:
        maximisation = False

    
    return ExploitationPhase(data, samples, maximisation, int(0.7*budget))

    

def main():
    dataPath = "data"
    resultsPath = "results/MyTool"

    for _ in range(10):
        results = {}
        budget = 100

        for file_name in os.listdir(dataPath):
            if file_name.endswith(".csv"):
                filePath = os.path.join(dataPath, file_name)
                outputFile = os.path.join(resultsPath, file_name)
                start_time = time.time()
                best_solution, best_performance = MySearch(filePath, budget)
                end_time = time.time()
                running_time = end_time - start_time

                results[file_name] = {
                    "Best Solution": best_solution,
                    "Best Performance": best_performance,
                }

                with open(outputFile, "a") as f:
                    if os.path.getsize(outputFile) == 0:
                        f.write("Best Performance,Running Time\n")
                    f.write("{},{}\n".format(best_performance, running_time))


        # Print the results
        for system, result in results.items():
            print(f"System: {system}")
            print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
            print(f"  Best Performance: {result['Best Performance']}")



if __name__ == "__main__":
    main()