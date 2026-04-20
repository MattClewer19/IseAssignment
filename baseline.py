import pandas as pd
import numpy as np
import os
import time

def randomSearch(filePath, budget):
    # read data into matrix
    data = pd.read_csv(filePath)

    # seperate feature columns and label
    configColumns = data.columns[:-1]
    performanceColumn = data.columns[-1]

    # decide if its a maximisation problem
    if os.path.basename(filePath).split('.')[0] == "---":
        maximisation = True
    else:
        maximisation = False

    # Extract the best and worst performance values
    if maximisation:
        worst_value = data[performanceColumn].min() / 2
    else:
        worst_value = data[performanceColumn].max() * 2 


    # Initialize the best solution and performance
    best_performance = -np.inf if maximisation else np.inf
    best_solution = []

    # Store all search results
    search_results = []

    for _ in range(budget):
        sampled_config = [int(np.random.choice(data[col].unique())) for col in configColumns]
        matched_row = data.loc[(data[configColumns] == pd.Series(sampled_config, index=configColumns)).all(axis=1)]

        if not matched_row.empty:
            # Existing configuration
            performance = matched_row[performanceColumn].iloc[0]
        else:
            # Non-existing configuration
            performance = worst_value
            
        # Update the best solution
        if maximisation:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config


    return [int(x) for x in best_solution], best_performance




def main():
    for _ in range(10):
        dataPath = "data"
        resultsPath = "results/Baseline"

        results = {}
        budget = 100

        for file_name in os.listdir(dataPath):
            if file_name.endswith(".csv"):
                filePath = os.path.join(dataPath, file_name)
                outputFile = os.path.join(resultsPath, file_name)
                start_time = time.time()
                best_solution, best_performance = randomSearch(filePath, budget)
                end_time = time.time()
                running_time = end_time - start_time
                results[file_name] = {
                    "Best Solution": best_solution,
                    "Best Performance": best_performance
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
