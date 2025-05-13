import os
import pandas as pd

# Constants
BASE_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/generated_answers'
MEDIA_TYPES = ['x', 'linkedin', 'press_release'] # Could add 'linkedin', 'press_release' back
RUN_NAMES = ['first', 'second', 'third', 'fourth']
SIZES = [10, 12, 75, 100]

def calculate_proportions(group_name, csv_files):
    # Concatenate all CSV files in the group
    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"File not found: {file}")
            return
            
    if not dfs:
        print(f"No data found for {group_name}")
        return
        
    data = pd.concat(dfs, ignore_index=True)

    metrics = ['relevance', 'attractiveness', 'fluency', 'style_strength']

    print(f"\nGroup: {group_name}")
    for metric in metrics:
        if metric not in data.columns:
            print(f"Metric '{metric}' not found in data columns.")
            continue
            
        total = data[metric].count()
        if total == 0:
            print(f"No data available for metric '{metric}' in group '{group_name}'.")
            continue
            
        distill_count = data[metric].sum()
        vanilla_count = total - distill_count
        distill_prop = distill_count / total * 100
        vanilla_prop = vanilla_count / total * 100
        
        print(f"\nMetric: {metric.capitalize()}")
        print(f"Distill chosen: {distill_count} times ({distill_prop:.2f}%)")
        print(f"Vanilla chosen: {vanilla_count} times ({vanilla_prop:.2f}%)")

def get_csv_files(media_type, size):
    files = []
    for run in RUN_NAMES:
        filename = f"generation_results_full_{run}_run_deterministic_evaluation_results_{size}.csv"
        filepath = os.path.join(BASE_DIR, media_type, filename)
        files.append(filepath)
    return files

def main():
    for media_type in MEDIA_TYPES:
        for size in SIZES:
            group_name = f"{media_type}_results_{size}"
            csv_files = get_csv_files(media_type, size)
            calculate_proportions(group_name, csv_files)

if __name__ == "__main__":
    main()
