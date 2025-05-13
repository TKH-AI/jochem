import os
import pandas as pd

# Constants
LONGT5_BASE_PATHS = {
    'x': "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_x/",
    'linkedin': "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_linkedin/",
    'press_release': "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_press_release/"
}

LLAMA_BASE_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/generated_answers'

MEDIA_TYPES = ['x', 'linkedin', 'press_release']
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

def get_llama_csv_files(media_type, size):
    files = []
    run_names = ['first', 'second', 'third', 'fourth']
    for run in run_names:
        filename = f"generation_results_full_{run}_run_deterministic_evaluation_results_{size}.csv"
        filepath = os.path.join(LLAMA_BASE_DIR, media_type, filename)
        files.append(filepath)
    return files

def get_longt5_csv_files(media_type, size):
    base_path = LONGT5_BASE_PATHS[media_type]
    files = []
    
    # Handle first 3 runs based on size
    if size in [75, 100]:
        if media_type == 'x':
            files.extend([
                os.path.join(base_path, "big", f"answers_all_runs_big_1.3_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_1.3_second_run_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_1.3_third_run_evaluation_results_{size}.csv")
            ])
        elif media_type == 'linkedin':
            files.extend([
                os.path.join(base_path, "big", f"answers_all_runs_big_1.2_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_1.2_second_run_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_1.2_third_run_evaluation_results_{size}.csv")
            ])
        elif media_type == 'press_release':
            files.extend([
                os.path.join(base_path, "big", f"answers_all_runs_big_1.2_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_second_run_1.2_evaluation_results_{size}.csv"),
                os.path.join(base_path, "big", f"answers_all_runs_big_third_run_1.2_evaluation_results_{size}.csv")
            ])
    else:  # size 10 or 12
        if media_type == 'x':
            files.extend([
                os.path.join(base_path, "small", f"answers_all_runs_small_1.3_more_sizes_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_1.3_second_run_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_1.3_third_run_evaluation_results_{size}.csv")
            ])
        elif media_type == 'linkedin':
            files.extend([
                os.path.join(base_path, "small", f"answers_all_runs_small_1.2_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_1.2_second_run_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_1.2_third_run_evaluation_results_{size}.csv")
            ])
        elif media_type == 'press_release':
            files.extend([
                os.path.join(base_path, "small", f"answers_all_runs_small_1.2_new_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_second_run_1.2_evaluation_results_{size}.csv"),
                os.path.join(base_path, "small", f"answers_all_runs_small_third_run_1.2_evaluation_results_{size}.csv")
            ])

    # Add fourth run - consistent naming across all media types and sizes
    fourth_run = f"answers_all_runs_big_and_small_fourth_run_goodone_evaluation_results_size_{size}.csv"
    files.append(os.path.join(base_path, fourth_run))
    
    return files

def main():
    print("=== LLaMA Results ===")
    for media_type in MEDIA_TYPES:
        for size in SIZES:
            group_name = f"llama_{media_type}_results_{size}"
            csv_files = get_llama_csv_files(media_type, size)
            calculate_proportions(group_name, csv_files)

    print("\n=== LongT5 Results ===")
    for media_type in MEDIA_TYPES:
        for size in SIZES:
            group_name = f"longt5_{media_type}_results_{size}"
            csv_files = get_longt5_csv_files(media_type, size)
            calculate_proportions(group_name, csv_files)

if __name__ == "__main__":
    main()