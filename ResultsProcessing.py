import pandas as pd
import os
from scipy.stats import mannwhitneyu

def analyse_file(path):
    data = pd.read_csv(path)  # change if needed

    return {
        "Best Performance": data["Best Performance"],
        "Running Time": data["Running Time"]
    }

def main():
    baseline_path = "results/Baseline"
    mytool_path = "results/MyTool"

    for file_name in os.listdir(baseline_path):
            if not file_name.endswith(".csv"):
                continue

            base_stats = analyse_file(os.path.join(baseline_path, file_name))
            tool_stats = analyse_file(os.path.join(mytool_path, file_name))

            baseline_perf = base_stats["Best Performance"]
            tool_perf = tool_stats["Best Performance"]

            _, p_value = mannwhitneyu(baseline_perf, tool_perf, alternative='two-sided')

            print(f"{file_name[:-4]} & {baseline_perf.mean():.4} & {tool_perf.mean():.4} & {'MyTool' if tool_perf.mean() < baseline_perf.mean() else 'Baseline'} & {p_value:.4} & {'Yes' if p_value < 0.05 else 'No'} \\\\")

    print("")
    print("=="*50)
    print("")

    for file_name in os.listdir(baseline_path):
            if not file_name.endswith(".csv"):
                continue

            base_stats = analyse_file(os.path.join(baseline_path, file_name))
            tool_stats = analyse_file(os.path.join(mytool_path, file_name))

            baseline_perf = base_stats["Running Time"]
            tool_perf = tool_stats["Running Time"]

            _, p_value = mannwhitneyu(baseline_perf, tool_perf, alternative='two-sided')

            print(f"{file_name[:-4]} & {baseline_perf.mean():.4} & {tool_perf.mean():.4} & {'MyTool' if tool_perf.mean() < baseline_perf.mean() else 'Baseline'} & {p_value:.4} & {'Yes' if p_value < 0.05 else 'No'} \\\\")


if __name__ == "__main__":
    main()