import os
import pandas as pd

# Define the paths to the directory containing the CSV files for X, LinkedIn, and Press Release
csv_path_x = "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_x/"
csv_path_linkedin = "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_linkedin/"
csv_path_press_release = "/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_press_release/"

# Define file paths for both small (low-resource) and big (high-resource) environments for X, LinkedIn, and Press Release
big_csv_files_x_results_75 = [
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_evaluation_results_75.csv"),
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_second_run_evaluation_results_75.csv"),
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_third_run_evaluation_results_75.csv")
]

big_csv_files_x_results_100 = [
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_evaluation_results_100.csv"),
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_second_run_evaluation_results_100.csv"),
    os.path.join(csv_path_x, "big/answers_all_runs_big_1.3_third_run_evaluation_results_100.csv")
]

small_csv_files_x_results_10 = [
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_more_sizes_evaluation_results_10.csv"),
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_second_run_evaluation_results_10.csv"),
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_third_run_evaluation_results_10.csv")
]

small_csv_files_x_results_12 = [
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_more_sizes_evaluation_results_12.csv"),
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_second_run_evaluation_results_12.csv"),
    os.path.join(csv_path_x, "small/answers_all_runs_small_1.3_third_run_evaluation_results_12.csv")
]

big_csv_files_linkedin_results_75 = [
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_evaluation_results_75.csv"),
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_second_run_evaluation_results_75.csv"),
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_third_run_evaluation_results_75.csv")
]

big_csv_files_linkedin_results_100 = [
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_evaluation_results_100.csv"),
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_second_run_evaluation_results_100.csv"),
    os.path.join(csv_path_linkedin, "big/answers_all_runs_big_1.2_third_run_evaluation_results_100.csv")
]

small_csv_files_linkedin_results_10 = [
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_evaluation_results_10.csv"),
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_second_run_evaluation_results_10.csv"),
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_third_run_evaluation_results_10.csv")
]

small_csv_files_linkedin_results_12 = [
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_evaluation_results_12.csv"),
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_second_run_evaluation_results_12.csv"),
    os.path.join(csv_path_linkedin, "small/answers_all_runs_small_1.2_third_run_evaluation_results_12.csv")
]

big_csv_files_press_release_results_75 = [
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_1.2_evaluation_results_75.csv"),
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_second_run_1.2_evaluation_results_75.csv"),
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_third_run_1.2_evaluation_results_75.csv")
]

big_csv_files_press_release_results_100 = [
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_1.2_evaluation_results_100.csv"),
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_second_run_1.2_evaluation_results_100.csv"),
    os.path.join(csv_path_press_release, "big/answers_all_runs_big_third_run_1.2_evaluation_results_100.csv")
]

small_csv_files_press_release_results_10 = [
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_1.2_new_evaluation_results_10.csv"),
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_second_run_1.2_evaluation_results_10.csv"),
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_third_run_1.2_evaluation_results_10.csv")
]

small_csv_files_press_release_results_12 = [
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_1.2_new_evaluation_results_12.csv"),
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_second_run_1.2_evaluation_results_12.csv"),
    os.path.join(csv_path_press_release, "small/answers_all_runs_small_third_run_1.2_evaluation_results_12.csv")
]

# Define a function to process the CSV files and calculate proportions
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

# Process each group
groups = {
    'big_csv_files_x_results_75': big_csv_files_x_results_75,
    'big_csv_files_x_results_100': big_csv_files_x_results_100,
    'small_csv_files_x_results_10': small_csv_files_x_results_10,
    'small_csv_files_x_results_12': small_csv_files_x_results_12,
    'big_csv_files_linkedin_results_75': big_csv_files_linkedin_results_75,
    'big_csv_files_linkedin_results_100': big_csv_files_linkedin_results_100,
    'small_csv_files_linkedin_results_10': small_csv_files_linkedin_results_10,
    'small_csv_files_linkedin_results_12': small_csv_files_linkedin_results_12,
    'big_csv_files_press_release_results_75': big_csv_files_press_release_results_75,
    'big_csv_files_press_release_results_100': big_csv_files_press_release_results_100,
    'small_csv_files_press_release_results_10': small_csv_files_press_release_results_10,
    'small_csv_files_press_release_results_12': small_csv_files_press_release_results_12
}

for group_name, csv_files in groups.items():
    calculate_proportions(group_name, csv_files)
