import logging
from tqdm import tqdm
import time
from datetime import datetime
import torch
import os
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from concurrent.futures import ProcessPoolExecutor

# Define paths for fine-tuned models
FINETUNED_MODELS = {
    'x': {
        'run1': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_x_run1/final',
        'run2': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_x_run2/final',
        'run3': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_x_run3/final',
        'run4': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_x_run4/final'
    },
    'linkedin': {
        'run1': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_linkedin_run1/final',
        'run2': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_linkedin_run2/final',
        'run3': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_linkedin_run3/final',
        'run4': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_linkedin_run4/final'
    },
    'press_release': {
        'run1': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_press_release_run1/final',
        'run2': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_press_release_run2/final',
        'run3': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_press_release_run3/final',
        'run4': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_press_release_run4/final'
    }
}

# Define paths for input data
BASE_DIR = '/projects/0/prjs1229/results_llama/generated_answers'
longt5_x_path = "/home/jheerebrugh/thesis-code-methods/results_x/generated_answers" 
longt5_linkedin_path = "/home/jheerebrugh/thesis-code-methods/results_linkedin/generated_answers"
longt5_press_path = "/home/jheerebrugh/thesis-code-methods/results_press_release/generated_answers"

# Define output directory  
output_dir = "/home/jheerebrugh/thesis-code-methods/results_x/perplexity_scores_finetuned"
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'perplexity_calculation_finetuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def calculate_perplexity(texts, model, tokenizer, device):
    # Calculate individual perplexities
    perplexities = []
    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, 
                            max_length=1024, padding=True)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities
    
def calculate_individual_perplexity(texts, model, tokenizer, device):
    """Calculate individual perplexity scores for each text"""
    perplexities = []
    
    for text in texts:
        try:
            if not isinstance(text, str) or not text.strip():
                perplexities.append(None)
                continue
                
            # Encode text with consistent max_length as training
            encodings = tokenizer(
                text.strip(),
                return_tensors='pt',
                truncation=True,
                max_length=1024,  # Match training max_length
                padding=True
            )
            input_ids = encodings.input_ids.to(device)
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                if 0 < perplexity < float('inf'):
                    perplexities.append(perplexity)
                else:
                    perplexities.append(None)
                    
        except Exception as e:
            logging.error(f"Error calculating individual perplexity: {str(e)}")
            perplexities.append(None)
            
    return perplexities

def init_finetuned_model(dataset_type, run_number):
    """Initialize fine-tuned model for specific dataset and run"""
    try:
        model_path = FINETUNED_MODELS[dataset_type][f'run{run_number}']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Loading fine-tuned model from {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        logging.error(f"Error initializing fine-tuned model for {dataset_type} run {run_number}: {str(e)}")
        raise

def process_file(input_file, model_type, run_type, env_type):
    """Optimized file processing"""
    model = None
    try:
        start_time = time.time()
        logging.info(f"Starting processing of {input_file}")
        
        # Read input file
        df = pd.read_csv(input_file)
        dataset_type = model_type.split('_')[1]
        
        if dataset_type == "press":
            dataset_type = "press_release"

        # Extract run number from run_type (e.g., 'run1' -> 1)
        run_number = int(run_type.replace('run', ''))
            
        # Initialize model without half precision to maintain accuracy
        model, tokenizer, device = init_finetuned_model(dataset_type, run_number)
        
        
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
                    logging.info(f"Processing columns for {percentage}%")
                    
                    # Prepare texts
                    distill_texts = df[distill_col].dropna().tolist()
                    vanilla_texts = df[vanilla_col].dropna().tolist()
                    
                    # Calculate individual perplexities
                    distill_perplexities = calculate_individual_perplexity(
                        distill_texts,
                        model,
                        tokenizer,
                        device
                    )
                    
                    vanilla_perplexities = calculate_individual_perplexity(
                        vanilla_texts,
                        model,
                        tokenizer,
                        device
                    )
                    
                    # Process results
                    valid_distill = [p for p in distill_perplexities if p is not None]
                    valid_vanilla = [p for p in vanilla_perplexities if p is not None]
                    
                    if valid_distill and valid_vanilla:
                        scores = {
                            'percentage': percentage,
                            'distill_perplexity': np.mean(valid_distill),
                            'vanilla_perplexity': np.mean(valid_vanilla),
                            'distill_std': np.std(valid_distill),
                            'vanilla_std': np.std(valid_vanilla),
                            'distill_count': len(valid_distill),
                            'vanilla_count': len(valid_vanilla)
                        }
                        results.append(scores)
                        logging.info(f"Completed {percentage}%: Distill={scores['distill_perplexity']:.2f}, "
                                   f"Vanilla={scores['vanilla_perplexity']:.2f}")
                    else:
                        logging.warning(f"No valid perplexity scores for {percentage}%")
                    
            except Exception as e:
                logging.error(f"Error processing percentage {percentage}%: {str(e)}")
                continue

        # Save results
        if results:
            results_df = pd.DataFrame(results)
            output_filename = f"perplexity_finetuned_{model_type}_{env_type}_{run_type}.csv"
            output_path = os.path.join(output_dir, output_filename)
            results_df.to_csv(output_path, index=False)
            
            end_time = time.time()
            logging.info(f"Saved perplexity scores to {output_path}")
            logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        else:
            logging.warning("No results were generated for any percentage")
            
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {str(e)}")
        raise e
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


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
        'press_release': {
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

    # Process LLaMa files in parallel
    # Define base paths for each dataset
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

    max_workers = min(torch.cuda.device_count() * 2, 8)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Process both LLaMA and LongT5 files for each dataset
        for dataset in ['x', 'linkedin', 'press_release']:
            for env in ['big', 'small']:
                # Process LLaMA files
                for filename, run in llama_files[dataset][env]:
                    input_path = os.path.join(base_paths[dataset]['llama'], filename)
                    futures.append(
                        executor.submit(
                            process_file, 
                            input_path, 
                            f"llama_{dataset}", 
                            run, 
                            env
                        )
                    )
                
                # Process LongT5 files
                for filename, run in longt5_files[dataset][env]:
                    input_path = os.path.join(base_paths[dataset]['longt5'], filename)
                    futures.append(
                        executor.submit(
                            process_file, 
                            input_path, 
                            f"longt5_{dataset}", 
                            run, 
                            env
                        )
                    )

        # Wait for all files to be processed
        for future in futures:
            future.result()

if __name__ == "__main__":
    # Enable cudnn benchmarking for faster conv operations
    torch.backends.cudnn.benchmark = True
    main()