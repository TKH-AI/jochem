import os
import pandas as pd
import numpy as np

def analyze_perplexity_scores():
    perplexity_dir = "/home/jheerebrugh/thesis-code-methods/results_x/perplexity_scores_finetuned"
    
    if not os.path.exists(perplexity_dir):
        print(f"Error: Directory {perplexity_dir} does not exist")
        return
    
    model_datasets = [
        'llama_x', 'llama_linkedin', 'llama_press_release',  
        'longt5_x', 'longt5_linkedin', 'longt5_press_release'
    ]
    runs = ['run1', 'run2', 'run3', 'run4']

    for model_dataset in model_datasets:
        model = model_dataset.split('_')[0]
        dataset = '_'.join(model_dataset.split('_')[1:])
        print(f"\n{'='*80}")
        print(f"Model: {model.upper()}, Dataset: {dataset.upper()}")
        print(f"{'='*80}")

        big_percentages = [12.5, 25.0, 50.0, 75.0, 100.0]
        small_percentages = [2.5, 5.0, 7.5, 10.0, 12.5]
        
        for env in ['big', 'small']:
            print(f"\n{env.upper()} Environment:")
            print(f"{'-'*60}")
            
            percentages = big_percentages if env == 'big' else small_percentages
            
            # Updated header for averaged scores
            print(f"{'%':<8} {'Distill':^12} {'Vanilla':^12}")
            print("-" * 40)
            
            for pct in percentages:
                distill_scores = []
                vanilla_scores = []
                
                for run in runs:
                    filename = f"perplexity_finetuned_{model_dataset}_{env}_{run}.csv"
                    filepath = os.path.join(perplexity_dir, filename)
                    
                    try:
                        if os.path.exists(filepath):
                            df = pd.read_csv(filepath)
                            row = df[df['percentage'] == pct]
                            if not row.empty:
                                distill_scores.append(row['distill_perplexity'].values[0])
                                vanilla_scores.append(row['vanilla_perplexity'].values[0])
                    except Exception as e:
                        print(f"Error reading {filename}: {str(e)}")
                        continue
                
                # Calculate and print averages if we have scores
                if distill_scores and vanilla_scores:
                    avg_distill = np.mean(distill_scores)
                    avg_vanilla = np.mean(vanilla_scores)
                    print(f"{pct:<8.1f} {avg_distill:12.2f} {avg_vanilla:12.2f}")
                    print("-" * 40)

if __name__ == "__main__":
    analyze_perplexity_scores()