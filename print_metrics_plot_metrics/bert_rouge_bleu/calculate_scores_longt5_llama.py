import os
import pandas as pd
import numpy as np
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import nltk
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch

# Ensure NLTK data is downloaded
nltk.download('punkt')

# Define paths
llama_path = "/home/jheerebrugh/thesis-code-methods/results_x/generated_answers/"
longt5_x_path = "/home/jheerebrugh/thesis-code-methods/results_x/generated_answers" 
longt5_linkedin_path = "/home/jheerebrugh/thesis-code-methods/results_linkedin/generated_answers"
longt5_press_path = "/home/jheerebrugh/thesis-code-methods/results_press_release/generated_answers"

# Define output directory
output_dir = "/home/jheerebrugh/thesis-code-methods/results_x/scores"
os.makedirs(output_dir, exist_ok=True)

def calculate_rouge_scores(labels, preds, rouge):
    """Calculate ROUGE scores in batches"""
    scores = [rouge.score(ref, pred) for ref, pred in zip(labels, preds)]
    return {
        'rouge1': np.mean([r['rouge1'].fmeasure for r in scores]),
        'rouge2': np.mean([r['rouge2'].fmeasure for r in scores]),
        'rougeL': np.mean([r['rougeL'].fmeasure for r in scores])
    }

def calculate_scores(df, distill_col, vanilla_col):
    """Calculate BERT, ROUGE and BLEU scores for a given dataframe and columns"""
    
    df_filtered = df.dropna(subset=['label', distill_col, vanilla_col])
    if df_filtered.empty:
        return None
        
    labels = df_filtered['label'].tolist()
    distill_preds = df_filtered[distill_col].tolist()
    vanilla_preds = df_filtered[vanilla_col].tolist()

    # Initialize Rouge scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Reduce batch size further and add error handling for BERT scores
    batch_size = 2  # Further reduced batch size
    max_retries = 3
    
    def try_bert_score(preds, refs):
        for retry in range(max_retries):
            try:
                # Clear CUDA cache before attempting
                torch.cuda.empty_cache()
                
                # Try smaller batch size if previous attempt failed
                current_batch_size = batch_size // (retry + 1)
                
                p, r, f1 = score(
                    preds, 
                    refs,
                    lang="en", 
                    verbose=False, 
                    batch_size=current_batch_size,
                    device='cuda:0'
                )
                return p, r, f1
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print(f"Failed to calculate BERT score after {max_retries} attempts")
                    raise e
                torch.cuda.empty_cache()
                continue
    
    try:
        # Calculate BERT scores with retry mechanism
        distill_p, distill_r, distill_f1 = try_bert_score(distill_preds, labels)
        vanilla_p, vanilla_r, vanilla_f1 = try_bert_score(vanilla_preds, labels)
    except Exception as e:
        print(f"Error calculating BERT scores: {str(e)}")
        # Return None or partial results if BERT fails
        return None

    # Clear CUDA cache after BERT calculations
    torch.cuda.empty_cache()

    # Pre-compute tokenization for BLEU
    tokenized_labels = [[ref.split()] for ref in labels]
    tokenized_distill = [pred.split() for pred in distill_preds]
    tokenized_vanilla = [pred.split() for pred in vanilla_preds]

    # Calculate BLEU scores
    distill_bleu = corpus_bleu(tokenized_labels, tokenized_distill)
    vanilla_bleu = corpus_bleu(tokenized_labels, tokenized_vanilla)

    # Calculate ROUGE scores
    distill_rouge = calculate_rouge_scores(labels, distill_preds, rouge)
    vanilla_rouge = calculate_rouge_scores(labels, vanilla_preds, rouge)

    scores = {
        'distill_bert_f1': distill_f1.mean().item(),
        'vanilla_bert_f1': vanilla_f1.mean().item(),
        'distill_bleu': distill_bleu,
        'vanilla_bleu': vanilla_bleu
    }

    # Add ROUGE scores
    scores.update({
        f'distill_{k}': v for k, v in distill_rouge.items()
    })
    scores.update({
        f'vanilla_{k}': v for k, v in vanilla_rouge.items()
    })
    
    return scores

def process_file(input_file, model_type, run_type, env_type):
    """Process a single CSV file and save scores"""
    try:
        print(f"Processing {input_file}...")
        df = pd.read_csv(input_file)
        
        # Add validation
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Define percentage mappings
        if env_type == "big":
            percentages = {12: 12.5, 25: 25.0, 50: 50.0, 75: 75.0, 100: 100.0}
        else:
            percentages = {2: 2.5, 5: 5.0, 7: 7.5, 10: 10.0, 12: 12.5}

        results = []
        for col_key, percentage in percentages.items():
            try:
                distill_col = f'distill_run_{col_key}'
                vanilla_col = f'vanilla_run_{col_key}'
                
                if distill_col in df.columns and vanilla_col in df.columns:
                    print(f"\nProcessing columns for {percentage}%:")
                    print(f"Distill column: {distill_col}")
                    print(f"Vanilla column: {vanilla_col}")
                    
                    non_null_count = df.dropna(subset=['label', distill_col, vanilla_col]).shape[0]
                    print(f"Number of valid rows: {non_null_count}")
                    
                    scores = calculate_scores(df, distill_col, vanilla_col)
                    if scores:
                        print(f"Scores for {percentage}%: {scores}")
                        scores['percentage'] = percentage
                        results.append(scores)
            except Exception as e:
                print(f"Error processing percentage {percentage}%: {str(e)}")
                continue

        if not results:
            print("Warning: No results generated!")
            return

        results_df = pd.DataFrame(results)
        print("\nFinal results:")
        print(results_df)
        
        output_filename = f"scores_{model_type}_{env_type}_{run_type}.csv"
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"Saved scores to {output_path}")
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        raise e


def main():
    # Define paths for each media type and run
    MEDIA_TYPES = ['x', 'linkedin', 'press_release']  
    BASE_DIR = '/projects/0/prjs1229/results_llama/generated_answers'  
    run_names = ['first', 'second', 'third', 'fourth']
    
    llama_files = {}
    
    for media_type in MEDIA_TYPES:
        llama_files[media_type] = {
            'big': [],
            'small': []
        }
        for run_name in run_names:
            csv_file = f"{BASE_DIR}/{media_type}/generation_results_full_{run_name}_run_deterministic.csv"
            llama_files[media_type]['big'].append((csv_file, f"run{run_names.index(run_name)+1}"))
            llama_files[media_type]['small'].append((csv_file, f"run{run_names.index(run_name)+1}"))

    longt5_files = {
        'x': {
            'big': [
                ("answers_all_runs_big_1.3.csv", "run1"),
                ("answers_all_runs_big_1.3_second_run.csv", "run2"),
                ("answers_all_runs_big_1.3_third_run.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ],
            'small': [
                ("answers_all_runs_small_1.3_more_sizes.csv", "run1"),
                ("answers_all_runs_small_1.3_second_run.csv", "run2"),
                ("answers_all_runs_small_1.3_third_run.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ]
        },
        'linkedin': {
            'big': [
                ("answers_all_runs_big_1.2.csv", "run1"),
                ("answers_all_runs_big_1.2_second_run.csv", "run2"),
                ("answers_all_runs_big_1.2_third_run.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ],
            'small': [
                ("answers_all_runs_small_1.2.csv", "run1"),
                ("answers_all_runs_small_1.2_second_run.csv", "run2"),
                ("answers_all_runs_small_1.2_third_run.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ]
        },
        'press': {
            'big': [
                ("answers_all_runs_big_1.2.csv", "run1"),
                ("answers_all_runs_big_second_run_1.2.csv", "run2"),
                ("answers_all_runs_big_third_run_1.2.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ],
            'small': [
                ("answers_all_runs_small_1.2_new.csv", "run1"),
                ("answers_all_runs_small_second_run_1.2.csv", "run2"),
                ("answers_all_runs_small_third_run_1.2.csv", "run3"),
                ("answers_all_runs_big_and_small_fourth_run_goodone.csv", "run4")
            ]
        }
    }

    # Base paths for both models
    base_paths = {
        'x': {
            'llama': BASE_DIR + '/x',
            'longt5': longt5_x_path 
        },
        'linkedin': {
            'llama': BASE_DIR + '/linkedin',
            'longt5': longt5_linkedin_path 
        },
        'press_release': {
            'llama': BASE_DIR + '/press_release',
            'longt5': longt5_press_path 
        }
    }

    # Process files in parallel for both models
    with ProcessPoolExecutor() as executor:
        futures = []
        
        # Process files for each dataset
        for dataset in MEDIA_TYPES:
            dataset_key = 'press' if dataset == 'press_release' else dataset
            
            for env in ['big', 'small']:
                # Process LLaMA files
                for filename, run in llama_files[dataset][env]:
                    input_path = os.path.join(base_paths[dataset]['llama'], filename)
                    print(f"Processing LLaMA file: {input_path}")
                    model_type = f"llama_{dataset.replace('_', '')}"
                    futures.append(
                        executor.submit(
                            process_file, 
                            input_path, 
                            model_type, 
                            run, 
                            env
                        )
                    )
                
                # Process LongT5 files
                for filename, run in longt5_files[dataset_key][env]:
                    input_path = os.path.join(base_paths[dataset]['longt5'], filename)
                    print(f"Processing LongT5 file: {input_path}")
                    model_type = f"longt5_{dataset.replace('_', '')}"
                    futures.append(
                        executor.submit(
                            process_file,
                            input_path,
                            model_type,
                            run,
                            env
                        )
                    )

    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()