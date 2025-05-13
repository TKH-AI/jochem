import os
import pandas as pd
import numpy as np

def load_and_aggregate_scores(base_dir, model_type):
    """Load and aggregate scores across runs for a specific model"""
    
    scores_list = []
    
    # Load scores from each run and environment
    for env in ['big', 'small']:
        for run in range(1, 5):  # 4 runs
            filename = f"scores_{model_type}_{env}_run{run}.csv"
            filepath = os.path.join(base_dir, filename)
            
            try:
                df = pd.read_csv(filepath)
                scores_list.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found - {filepath}")
                continue

    if not scores_list:
        print(f"No scores found for {model_type}")
        return None

    # Concatenate all runs
    all_runs = pd.concat(scores_list)
    
    # Group by percentage and calculate mean across runs
    avg_scores = all_runs.groupby('percentage').mean()
    
    return avg_scores

def print_formatted_results(scores_df, model_name):
    """Print formatted results for a given model"""
    
    print(f"\n{'='*80}")
    print(f"Results for {model_name}")
    print(f"{'='*80}")
    
    # Specific percentages to print
    percentages = [2.5, 5.0, 7.5, 10.0, 12.5, 25.0, 50.0, 75.0, 100.0]
    
    for percentage in percentages:
        try:
            print(f"\nPercentage: {percentage}%")
            print("-" * 40)
            
            # BERT Score
            print(f"BERT Score:")
            print(f"  - Distill: {scores_df.loc[percentage, 'distill_bert_f1']:.4f}")
            print(f"  - Vanilla: {scores_df.loc[percentage, 'vanilla_bert_f1']:.4f}")
            
            # BLEU Score
            print(f"BLEU Score:")
            print(f"  - Distill: {scores_df.loc[percentage, 'distill_bleu']:.4f}")
            print(f"  - Vanilla: {scores_df.loc[percentage, 'vanilla_bleu']:.4f}")
            
            # ROUGE Scores
            print(f"ROUGE Scores:")
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                print(f"  {metric}:")
                print(f"    - Distill: {scores_df.loc[percentage, f'distill_{metric}']:.4f}")
                print(f"    - Vanilla: {scores_df.loc[percentage, f'vanilla_{metric}']:.4f}")
        except KeyError:
            continue

def main():
    # Define base directory where scores are saved
    base_dir = "/home/jheerebrugh/thesis-code-methods/results_x/scores"
    
    # Define models and media types 
    models = {
        'llama': ['x', 'linkedin', 'pressrelease'],
        'longt5': ['x', 'linkedin', 'press']
    }
    
    # Process each model type and media
    for model, media_types in models.items():
        for media in media_types:
            model_type = f"{model}_{media}"
            
            # Load and process scores
            scores = load_and_aggregate_scores(base_dir, model_type)
            if scores is not None:
                print_formatted_results(scores, model_type)

if __name__ == "__main__":
    main()
