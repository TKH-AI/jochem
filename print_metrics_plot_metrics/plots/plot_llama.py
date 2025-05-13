import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Constants from original file
MEDIA_TYPES = {'x': 'x', 'linkedin': 'linkedin', 'press_release': 'pressrelease'}
BASE_DIR = '/projects/0/prjs1229/results_llama/generated_answers'
OUTPUT_DIR = '/projects/0/prjs1229/results_llama/plots'
SCORES_DIR = '/home/jheerebrugh/thesis-code-methods/results_x/scores'

# Set plotting style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", 
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10
})

def load_scores(media_type, run_number, environment):
    """Load pre-calculated scores from CSV files"""
    if media_type == 'press_release':
        media_name = 'pressrelease'
    else:
        media_name = media_type
    
    env_name = environment if environment == 'small' else 'big'
    
    filename = f"scores_llama_{media_name}_{env_name}_run{run_number}.csv"
    filepath = os.path.join(SCORES_DIR, filename)
    
    try:
        df = pd.read_csv(filepath)
        metrics = {'run': f'Run {run_number}', 'distill': {}, 'vanilla': {}}
        
        for _, row in df.iterrows():
            percentage = float(row['percentage'])
            metrics['distill'][percentage] = row['distill_bert_f1']
            metrics['vanilla'][percentage] = row['vanilla_bert_f1']
            
        return metrics
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def process_files_individual(media_type, environment):
    """Load pre-calculated scores for all runs of a media type"""
    all_run_metrics = []
    valid_data = False
    
    for run_idx in range(1, 5):
        metrics = load_scores(media_type, run_idx, environment)
        if metrics:
            all_run_metrics.append(metrics)
            valid_data = True
            
    if not valid_data:
        print(f"Warning: No valid data found for {media_type} in {environment} environment")
        return []
        
    return all_run_metrics

def plot_combined_horizontal(all_metrics_small, all_metrics_large):
    """Plot combined small and large environment results horizontally"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    train_sizes_small = [2.5, 5.0, 7.5, 10.0, 12.5]
    train_sizes_large = [25.0, 50.0, 75.0, 100.0]
    train_sizes = train_sizes_small + train_sizes_large
    
    colors = {'distill': '#1f77b4', 'vanilla': '#ff7f0e'}
    
    for i, media_type in enumerate(MEDIA_TYPES):
        title = 'X' if media_type == 'x' else ('LinkedIn' if media_type == 'linkedin' else 'Press Release')
        
        final_vanilla_value = None
        
        for model in ['distill', 'vanilla']:
            values_small = []
            std_devs_small = []
            
            # Process small environment if data exists
            if media_type in all_metrics_small and all_metrics_small[media_type]:
                for size in train_sizes_small:
                    vals = [run[model][size] for run in all_metrics_small[media_type] if run and model in run and size in run[model]]
                    if vals:
                        values_small.append(np.mean(vals))
                        std_devs_small.append(np.std(vals))
                    else:
                        values_small.append(np.nan)
                        std_devs_small.append(np.nan)
            
            values_large = []
            std_devs_large = []
            
            # Process large environment if data exists
            if media_type in all_metrics_large and all_metrics_large[media_type]:
                for size in train_sizes_large:
                    vals = [run[model][size] for run in all_metrics_large[media_type] if run and model in run and size in run[model]]
                    if vals:
                        values_large.append(np.mean(vals))
                        std_devs_large.append(np.std(vals))
                    else:
                        values_large.append(np.nan)
                        std_devs_large.append(np.nan)
            
            values = values_small + values_large
            std_devs = std_devs_small + std_devs_large
            
            valid_indices = ~np.isnan(values)
            if np.any(valid_indices):
                valid_sizes = np.array(train_sizes)[valid_indices]
                valid_values = np.array(values)[valid_indices]
                valid_std_devs = np.array(std_devs)[valid_indices]
                
                marker = 'o' if model == 'distill' else '^'
                axes[i].plot(valid_sizes, valid_values,
                           marker=marker,
                           label=model.capitalize(),
                           linestyle='-',
                           linewidth=2,
                           color=colors[model])
                           
                axes[i].fill_between(valid_sizes,
                                   valid_values - valid_std_devs,
                                   valid_values + valid_std_devs,
                                   alpha=0.2,
                                   color=colors[model])
                                   
                if model == 'vanilla':
                    # Store the final vanilla value at 100%
                    final_idx = np.where(valid_sizes == 100.0)[0]
                    if len(final_idx) > 0:
                        final_vanilla_value = valid_values[final_idx[0]]
        
        # Add horizontal line at final vanilla value
        if final_vanilla_value is not None:
            axes[i].axhline(y=final_vanilla_value,
                          color='#2ca02c',
                          linestyle=':',
                          linewidth=2,
                          label='Vanilla @ 100\\%')
                                   
        axes[i].set_title(f'{title}', fontsize=16)
        axes[i].set_xlabel('Training set size (\\% of full dataset)', fontsize=14)
        if i == 0:
            axes[i].set_ylabel('BERT F1', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.0),
              ncol=3,
              fontsize=20)
              
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_horizontal_bert_f1.png'),
                dpi=300)
    plt.close()

def plot_combined_vertical(all_metrics_small, all_metrics_large):
    """Plot combined small and large environment results vertically"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    train_sizes_small = [2.5, 5.0, 7.5, 10.0, 12.5]
    train_sizes_large = [25.0, 50.0, 75.0, 100.0]
    train_sizes = train_sizes_small + train_sizes_large
    
    colors = {'distill': '#1f77b4', 'vanilla': '#ff7f0e'}
    
    for i, media_type in enumerate(MEDIA_TYPES):
        title = 'X' if media_type == 'x' else ('LinkedIn' if media_type == 'linkedin' else 'Press Release')
        
        final_vanilla_value = None
        
        for model in ['distill', 'vanilla']:
            # Process small environment
            values_small = []
            std_devs_small = []
            for size in train_sizes_small:
                vals = [run[model][size] for run in all_metrics_small[media_type] if run and size in run[model]]
                if vals:
                    values_small.append(np.mean(vals))
                    std_devs_small.append(np.std(vals))
                else:
                    values_small.append(np.nan)
                    std_devs_small.append(np.nan)
                    
            # Process large environment
            values_large = []
            std_devs_large = []
            for size in train_sizes_large:
                vals = [run[model][size] for run in all_metrics_large[media_type] if run and size in run[model]]
                if vals:
                    values_large.append(np.mean(vals))
                    std_devs_large.append(np.std(vals))
                else:
                    values_large.append(np.nan)
                    std_devs_large.append(np.nan)
                    
            values = values_small + values_large
            std_devs = std_devs_small + std_devs_large
            
            valid_indices = ~np.isnan(values)
            if np.any(valid_indices):
                valid_sizes = np.array(train_sizes)[valid_indices]
                valid_values = np.array(values)[valid_indices]
                valid_std_devs = np.array(std_devs)[valid_indices]
                
                marker = 'o' if model == 'distill' else '^'
                axes[i].plot(valid_sizes, valid_values,
                           marker=marker,
                           label=model.capitalize(),
                           linestyle='-',
                           linewidth=2,
                           color=colors[model])
                           
                axes[i].fill_between(valid_sizes,
                                   valid_values - valid_std_devs,
                                   valid_values + valid_std_devs,
                                   alpha=0.2,
                                   color=colors[model])
                                   
                if model == 'vanilla':
                    # Store the final vanilla value at 100%
                    final_idx = np.where(valid_sizes == 100.0)[0]
                    if len(final_idx) > 0:
                        final_vanilla_value = valid_values[final_idx[0]]
        
        # Add horizontal line at final vanilla value
        if final_vanilla_value is not None:
            axes[i].axhline(y=final_vanilla_value,
                          color='#2ca02c',
                          linestyle=':',
                          linewidth=2,
                          label='Vanilla @ 100\\%')
                                   
        axes[i].set_title(f'{title}', fontsize=16)
        axes[i].set_xlabel('Training set size (\\% of full dataset)', fontsize=14)
        axes[i].set_ylabel('BERT F1', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        
    # Move legend outside plots for vertical layout
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='center',
              bbox_to_anchor=(0.5, 0.98),
              ncol=3,
              fontsize=20)
              
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_vertical_bert_f1.png'),
                dpi=300)
    plt.close()

def main():
    # Load metrics for both environments
    all_metrics_small = {}
    all_metrics_large = {}
    
    for media_type in MEDIA_TYPES:
        print(f"Processing {media_type}...")
        small_metrics = process_files_individual(media_type, 'small')
        large_metrics = process_files_individual(media_type, 'large')
        
        if small_metrics:  # Only add if we have valid data
            all_metrics_small[media_type] = small_metrics
        if large_metrics:  # Only add if we have valid data
            all_metrics_large[media_type] = large_metrics
    
    if not all_metrics_small and not all_metrics_large:
        print("Error: No valid data found for any media type")
        return
    
    # Generate both plot layouts
    plot_combined_horizontal(all_metrics_small, all_metrics_large)
    plot_combined_vertical(all_metrics_small, all_metrics_large)
    
    print(f"All plots have been saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()