import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define paths  
scores_dir = "/home/jheerebrugh/thesis-code-methods/results_x/scores"
plots_dir = "/home/jheerebrugh/thesis-code-methods/plots_x/model_comparison"
os.makedirs(plots_dir, exist_ok=True)

# Set plotting style
plt.rcParams.update({
    "text.usetex": True, 
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14
})

def load_and_process_scores(model_type, dataset, env_type):
    """Load scores from all runs and calculate overall means and std"""
    all_runs_data = []
    
    # Load data from all runs
    for run in range(1, 5):
        filename = f"scores_{model_type}_{dataset}_{env_type}_run{run}.csv"
        filepath = os.path.join(scores_dir, filename)
        try:
            df = pd.read_csv(filepath)
            # Add run number for tracking
            df['run'] = run
            all_runs_data.append(df)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            continue
    
    if not all_runs_data:
        return None
        
    # Combine all runs
    combined_df = pd.concat(all_runs_data, ignore_index=True)
    
    # Calculate means and std across runs for each percentage
    metrics = ['distill_bert_f1', 'vanilla_bert_f1']
    results = {}
    
    for metric in metrics:
        # Calculate mean and std for each percentage across runs
        stats = combined_df.groupby('percentage')[metric].agg(['mean', 'std']).reset_index()
        results[metric] = stats
        
    return results

def create_comparison_plot(env_type, method_type):
    """Create comparison plot for specific environment and method type (distill/vanilla)"""
    # Load and process data for each model and dataset
    llama_x_stats = load_and_process_scores("llama", "x", env_type)
    llama_linkedin_stats = load_and_process_scores("llama", "linkedin", env_type)
    llama_press_stats = load_and_process_scores("llama", "press", env_type)
    
    longt5_x_stats = load_and_process_scores("longt5", "x", env_type)
    longt5_linkedin_stats = load_and_process_scores("longt5", "linkedin", env_type)
    longt5_press_stats = load_and_process_scores("longt5", "press", env_type)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define datasets for comparison
    comparisons = [
        ('X Dataset', llama_x_stats, longt5_x_stats, 0),
        ('LinkedIn Dataset', llama_linkedin_stats, longt5_linkedin_stats, 1),
        ('Press Release Dataset', llama_press_stats, longt5_press_stats, 2)
    ]

    bert_col = f'{method_type}_bert_f1'

    for title, llama_data, longt5_data, idx in comparisons:
        if llama_data is None or longt5_data is None:
            continue
            
        # Plot LLaMA
        llama_stats = llama_data[bert_col]
        axes[idx].plot(llama_stats['percentage'], llama_stats['mean'], 
                      marker='o', label='LLaMA', linewidth=2)
        axes[idx].fill_between(llama_stats['percentage'],
                             llama_stats['mean'] - llama_stats['std'],
                             llama_stats['mean'] + llama_stats['std'],
                             alpha=0.2)

        # Plot LongT5
        longt5_stats = longt5_data[bert_col]
        axes[idx].plot(longt5_stats['percentage'], longt5_stats['mean'], 
                      marker='^', label='LongT5', linewidth=2)
        axes[idx].fill_between(longt5_stats['percentage'],
                             longt5_stats['mean'] - longt5_stats['std'],
                             longt5_stats['mean'] + longt5_stats['std'],
                             alpha=0.2)

        # Add final point reference line
        max_percentage = 12.5 if env_type == 'small' else 100.0
        final_mean = llama_stats[llama_stats['percentage'] == max_percentage]['mean'].values[0]
        axes[idx].axhline(y=final_mean, color='green', linestyle=':', linewidth=2, 
                         label=f'Final point @ {max_percentage}\\%')

        # Customize subplot
        axes[idx].set_xlabel('Training set size (\\% of full dataset)')
        if idx == 0:
            axes[idx].set_ylabel('BERT F1')
        axes[idx].set_title(title)
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].legend()

    # Set overall title
    plt.suptitle(f'{env_type.capitalize()} Environment - {method_type.capitalize()} Method', y=1.05)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f'{env_type}_{method_type}_model_comparison.png'), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    
def main():
    # Create separate plots for each combination
    environments = ['big', 'small']
    methods = ['distill', 'vanilla']
    
    for env in environments:
        for method in methods:
            create_comparison_plot(env, method)
            print(f"Created plot for {env} environment, {method} method")
    
    print(f"All comparison plots have been saved to {plots_dir}")

if __name__ == "__main__":
    main()