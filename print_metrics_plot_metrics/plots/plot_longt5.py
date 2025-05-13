# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from bert_score import score
# import numpy as np
# import nltk

# # Ensure NLTK data is downloaded
# nltk.download('punkt')

# # Define the output directory for plots
# output_dir = "/home/jheerebrugh/thesis-code-methods/plots_x/test_generated"
# os.makedirs(output_dir, exist_ok=True)

# # Define the paths to the directory containing the CSV files for X, LinkedIn, and Press Release
# csv_path_x = "/home/jheerebrugh/thesis-code-methods/results_x/generated_answers"
# csv_path_linkedin = "/home/jheerebrugh/thesis-code-methods/results_linkedin/generated_answers"
# csv_path_press_release = "/home/jheerebrugh/thesis-code-methods/results_press_release/generated_answers"

# # Update file paths to include fourth run
# big_csv_files_x = [
#     os.path.join(csv_path_x, "answers_all_runs_big_1.3.csv"),
#     os.path.join(csv_path_x, "answers_all_runs_big_1.3_second_run.csv"), 
#     os.path.join(csv_path_x, "answers_all_runs_big_1.3_third_run.csv"),
#     os.path.join(csv_path_x, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# small_csv_files_x = [
#     os.path.join(csv_path_x, "answers_all_runs_small_1.3_more_sizes.csv"),
#     os.path.join(csv_path_x, "answers_all_runs_small_1.3_second_run.csv"),
#     os.path.join(csv_path_x, "answers_all_runs_small_1.3_third_run.csv"),
#     os.path.join(csv_path_x, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# big_csv_files_linkedin = [
#     os.path.join(csv_path_linkedin, "answers_all_runs_big_1.2.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_big_1.2_second_run.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_big_1.2_third_run.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# small_csv_files_linkedin = [
#     os.path.join(csv_path_linkedin, "answers_all_runs_small_1.2.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_small_1.2_second_run.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_small_1.2_third_run.csv"),
#     os.path.join(csv_path_linkedin, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# big_csv_files_press_release = [
#     os.path.join(csv_path_press_release, "answers_all_runs_big_1.2.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_big_second_run_1.2.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_big_third_run_1.2.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# small_csv_files_press_release = [
#     os.path.join(csv_path_press_release, "answers_all_runs_small_1.2_new.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_small_second_run_1.2.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_small_third_run_1.2.csv"),
#     os.path.join(csv_path_press_release, "answers_all_runs_big_and_small_fourth_run_goodone.csv")
# ]

# # Modify mapping for big environment (removed 5%)
# percentage_mapping_big = {12: 12.5, 25: 25.0, 50: 50.0, 75: 75.0, 100: 100.0}
# percentage_mapping_small = {2: 2.5, 5: 5.0, 7: 7.5, 10: 10.0, 12: 12.5}

# # Set LaTeX rendering and Computer Modern font
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern"],
#     "axes.labelsize": 14,      # Axis labels
#     "axes.titlesize": 18,      # Title size
#     "xtick.labelsize": 12,     # X-axis tick labels
#     "ytick.labelsize": 12,     # Y-axis tick labels
#     "legend.fontsize": 14      # Legend font size
# })

# # Modified process_files function to handle combined fourth run file
# def process_files(csv_files, percentage_mapping, train_sizes, models):
#     all_metrics = {model: {size: [] for size in train_sizes} for model in models}

#     for run_idx, csv_file in enumerate(csv_files, 1):
#         input_file_path = csv_file
#         print(f"\nProcessing Run {run_idx}: {csv_file}")
        
#         try:
#             df = pd.read_csv(input_file_path)
#         except FileNotFoundError:
#             print(f"Error: File {csv_file} not found. Skipping this run.")
#             continue

#         # Special handling for fourth run file that contains all sizes
#         is_fourth_run = "fourth_run_goodone" in csv_file
        
#         for column_key, actual_percentage in percentage_mapping.items():
#             # For fourth run, only process relevant sizes based on environment
#             if is_fourth_run and not (column_key in df.columns or f'distill_run_{column_key}' in df.columns):
#                 continue

#             distill_col = f'distill_run_{column_key}' if not is_fourth_run else f'distill_run_{column_key}'
#             vanilla_col = distill_col.replace('distill', 'vanilla')

#             if distill_col not in df.columns or vanilla_col not in df.columns:
#                 print(f"Warning: Missing expected columns '{distill_col}' or '{vanilla_col}' in {csv_file}")
#                 continue

#             df_filtered = df.dropna(subset=['label', distill_col, vanilla_col])
#             if df_filtered.empty:
#                 continue

#             labels = df_filtered['label'].tolist()
#             distill_preds = df_filtered[distill_col].tolist()
#             vanilla_preds = df_filtered[vanilla_col].tolist()

#             # Calculate BERT F1 Scores
#             distill_bert_precision, distill_bert_recall, distill_bert_f1 = score(distill_preds, labels, lang="en", verbose=False)
#             vanilla_bert_precision, vanilla_bert_recall, vanilla_bert_f1 = score(vanilla_preds, labels, lang="en", verbose=False)

#             avg_distill_bert_f1 = distill_bert_f1.mean().item()
#             avg_vanilla_bert_f1 = vanilla_bert_f1.mean().item()

#             all_metrics['distill'][actual_percentage].append(avg_distill_bert_f1)
#             all_metrics['vanilla'][actual_percentage].append(avg_vanilla_bert_f1)

#     return all_metrics

# # Plotting function for BERT F1 Scores with standard deviation as shaded areas
# def plot_bert_f1(metrics_x, metrics_linkedin, metrics_press_release, train_sizes, models, output_dir, environment, final_point_small=12.5, final_point_big=100):
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create three subplots side by side

#     for i, (metrics, title) in enumerate(zip([metrics_x, metrics_linkedin, metrics_press_release], ['X', 'LinkedIn', 'Press Release'])):
#         for model in models:
#             marker = 'o' if model == 'distill' else '^'  # Triangle for vanilla
#             values = [np.mean(metrics[model][size]) for size in train_sizes if metrics[model][size]]
#             std_devs = [np.std(metrics[model][size]) for size in train_sizes if metrics[model][size]]  # Standard deviations
#             axes[i].plot(train_sizes, values, marker=marker, label=model.capitalize(), linestyle='-', linewidth=2)
#             # Shaded area for standard deviation
#             axes[i].fill_between(train_sizes, 
#                                  np.array(values) - np.array(std_devs), 
#                                  np.array(values) + np.array(std_devs), 
#                                  alpha=0.2)
        
#         # Add horizontal green dotted line for vanilla final point
#         final_point = final_point_small if environment == 'small' else final_point_big
#         axes[i].axhline(y=values[-1], color='green', linestyle=':', linewidth=2, label=f'Vanilla @ {final_point}\\%')

#         axes[i].set_title(f'{title}', fontsize=16)

#         axes[i].set_xlabel('Training set size (\\% of full dataset)', fontsize=14)
#         if i == 0:
#             axes[i].set_ylabel('BERT F1', fontsize=14)
#         axes[i].tick_params(axis='both', which='major', labelsize=12)
#         axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Adjust the legend
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=20)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to fit the larger legend
#     plt.savefig(os.path.join(output_dir, f'{environment}_env_bert_f1_x_linkedin_press_release.png'), dpi=300)
#     plt.close()

# # Process X, LinkedIn, and Press Release datasets
# small_metrics_x = process_files(small_csv_files_x, percentage_mapping_small, list(percentage_mapping_small.values()), ['distill', 'vanilla'])
# big_metrics_x = process_files(big_csv_files_x, percentage_mapping_big, list(percentage_mapping_big.values()), ['distill', 'vanilla'])

# small_metrics_linkedin = process_files(small_csv_files_linkedin, percentage_mapping_small, list(percentage_mapping_small.values()), ['distill', 'vanilla'])
# big_metrics_linkedin = process_files(big_csv_files_linkedin, percentage_mapping_big, list(percentage_mapping_big.values()), ['distill', 'vanilla'])

# small_metrics_press_release = process_files(small_csv_files_press_release, percentage_mapping_small, list(percentage_mapping_small.values()), ['distill', 'vanilla'])
# big_metrics_press_release = process_files(big_csv_files_press_release, percentage_mapping_big, list(percentage_mapping_big.values()), ['distill', 'vanilla'])

# # Plot results for small and big environments (including Press Release)
# plot_bert_f1(small_metrics_x, small_metrics_linkedin, small_metrics_press_release, list(percentage_mapping_small.values()), ['distill', 'vanilla'], output_dir, "small", final_point_small=12.5)
# plot_bert_f1(big_metrics_x, big_metrics_linkedin, big_metrics_press_release, list(percentage_mapping_big.values()), ['distill', 'vanilla'], output_dir, "big", final_point_big=100)

# print(f"BERT F1 plots for X, LinkedIn, and Press Release datasets (small and big environments) have been saved to {output_dir}")

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set LaTeX rendering and Computer Modern font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,      # Axis labels
    "axes.titlesize": 18,      # Title size
    "xtick.labelsize": 12,     # X-axis tick labels
    "ytick.labelsize": 12,     # Y-axis tick labels
    "legend.fontsize": 14      # Legend font size
})

# Define paths
scores_dir = "/home/jheerebrugh/thesis-code-methods/results_x/scores"
output_dir = "/home/jheerebrugh/thesis-code-methods/plots_x/test_generated"
os.makedirs(output_dir, exist_ok=True)

def process_scores(env_type, dataset):
    """Process scores from all runs for a specific environment and dataset"""
    metrics = {'distill': {}, 'vanilla': {}}
    
    for run in range(1, 5):  # Process runs 1-4
        filename = f"scores_longt5_{dataset}_{env_type}_run{run}.csv"
        filepath = os.path.join(scores_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            for percentage in df['percentage'].unique():
                if percentage not in metrics['distill']:
                    metrics['distill'][percentage] = []
                    metrics['vanilla'][percentage] = []
                
                row = df[df['percentage'] == percentage].iloc[0]
                metrics['distill'][percentage].append(row['distill_bert_f1'])
                metrics['vanilla'][percentage].append(row['vanilla_bert_f1'])
        except FileNotFoundError:
            print(f"Warning: File {filename} not found")
            continue
    
    return metrics

def plot_bert_f1(metrics_x, metrics_linkedin, metrics_press_release, environment, final_point_small=12.5, final_point_big=100):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = {
        0: ('x', metrics_x, 'X'),
        1: ('linkedin', metrics_linkedin, 'LinkedIn'),
        2: ('press_release', metrics_press_release, 'Press Release')
    }
    
    for i, (dataset, metrics, title) in datasets.items():
        train_sizes = sorted(metrics['distill'].keys())
        
        for model in ['distill', 'vanilla']:
            marker = 'o' if model == 'distill' else '^'
            values = [np.mean(metrics[model][size]) for size in train_sizes if metrics[model][size]]
            std_devs = [np.std(metrics[model][size]) for size in train_sizes if metrics[model][size]]
            
            axes[i].plot(train_sizes, values, marker=marker, label=model.capitalize(), linestyle='-', linewidth=2)
            axes[i].fill_between(train_sizes,
                               np.array(values) - np.array(std_devs),
                               np.array(values) + np.array(std_devs),
                               alpha=0.2)
        
        final_point = final_point_small if environment == 'small' else final_point_big
        axes[i].axhline(y=values[-1], color='green', linestyle=':', linewidth=2, label=f'Vanilla @ {final_point}\\%')
        
        axes[i].set_title(f'{title}', fontsize=16)
        axes[i].set_xlabel('Training set size (\\% of full dataset)', fontsize=14)
        if i == 0:
            axes[i].set_ylabel('BERT F1', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, f'{environment}_env_bert_f1_x_linkedin_press_release.png'), dpi=300)
    plt.close()

def print_metrics_table(metrics_x, metrics_linkedin, metrics_press, environment):
    """Print BERT F1 scores and standard deviations in a machine-readable format"""
    
    datasets = {
        'X': metrics_x,
        'LinkedIn': metrics_linkedin,
        'Press Release': metrics_press
    }
    
    print(f"\nMETRICS_{environment.upper()}_START")
    
    for title, metrics in datasets.items():
        print(f"DATASET: {title}")
        
        train_sizes = sorted(metrics['distill'].keys())
        for size in train_sizes:
            print(f"SIZE: {size}")
            
            if metrics['distill'][size] and metrics['vanilla'][size]:
                distill_mean = np.mean(metrics['distill'][size])
                distill_std = np.std(metrics['distill'][size])
                vanilla_mean = np.mean(metrics['vanilla'][size])
                vanilla_std = np.std(metrics['vanilla'][size])
                
                print(f"DISTILL: mean={distill_mean:.4f} std={distill_std:.4f}")
                print(f"VANILLA: mean={vanilla_mean:.4f} std={vanilla_std:.4f}")
            else:
                print("DISTILL: NO_DATA")
                print("VANILLA: NO_DATA")
                
    print(f"METRICS_{environment.upper()}_END\n")

def main():
    # Process scores for small environment
    small_metrics_x = process_scores('small', 'x')
    small_metrics_linkedin = process_scores('small', 'linkedin')
    small_metrics_press = process_scores('small', 'press')
    
    # Process scores for big environment
    big_metrics_x = process_scores('big', 'x')
    big_metrics_linkedin = process_scores('big', 'linkedin')
    big_metrics_press = process_scores('big', 'press')
    
    # Generate plots and print metrics
    plot_bert_f1(small_metrics_x, small_metrics_linkedin, small_metrics_press, 'small')
    print_metrics_table(small_metrics_x, small_metrics_linkedin, small_metrics_press, 'small')
    
    plot_bert_f1(big_metrics_x, big_metrics_linkedin, big_metrics_press, 'big')
    print_metrics_table(big_metrics_x, big_metrics_linkedin, big_metrics_press, 'big')
    
    print(f"BERT F1 plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()