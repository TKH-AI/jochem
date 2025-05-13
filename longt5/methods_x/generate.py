from transformers import pipeline, AutoTokenizer, LongT5ForConditionalGeneration
from typing import Dict, Any
from datasets import Dataset
import pandas as pd
import os
import torch
import warnings
import logging

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths and directories
base_path = "/home/jheerebrugh/thesis-code-methods/results_x"
test_file_path = "/home/jheerebrugh/thesis-code-methods/resources_x_4/test_dataset.csv"
run_sizes = ['run_2', 'run_5', 'run_7', 'run_10', 'run_12', 'run_25', 'run_50', 'run_75', 'run_100']

device = 0 if torch.cuda.is_available() else -1

# Vanilla and distill output directories
distill_base_path = f"{base_path}/distill"
vanilla_base_path = f"{base_path}/vanilla"
output_file_path = f"{base_path}/generated_answers/answers_all_runs_big_and_small_fourth_run_goodone.csv"

# Create output directory if it doesn't exist
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"Created output directory: {output_dir}")

# Load the test dataset
df = pd.read_csv(test_file_path)
dataset = Dataset.from_pandas(df)

# Function to get the latest checkpoint from a directory
def get_latest_checkpoint(run_path):
    checkpoints = [ckpt for ckpt in os.listdir(run_path) if ckpt.startswith('checkpoint-')]
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    return os.path.join(run_path, latest_checkpoint)

def generate_answers_batch(samples: Dataset, distill_generators: Dict[str, Any], vanilla_generators: Dict[str, Any], batch_size: int = 32) -> Dataset:
    total_samples = len(samples)
    samples_dict = samples.to_dict()
    
    # Convert inputs once
    inputs_dataset = Dataset.from_dict({
        "text": ["predict: " + text for text in samples["input"]]
    })
    
    for run_size in run_sizes:
        logger.info(f"Processing {run_size}")
        
        # Initialize new columns
        samples_dict[f'distill_{run_size}'] = [''] * total_samples
        samples_dict[f'vanilla_{run_size}'] = [''] * total_samples
        
        # Process with distill model
        try:
            outputs = distill_generators[run_size](
                inputs_dataset["text"],
                batch_size=batch_size,
                num_beams=1,
                do_sample=False,
                max_new_tokens=256  # Add specific max_length
            )
            
            # Update the dataset
            for idx, output in enumerate(outputs):
                if idx < total_samples:
                    samples_dict[f'distill_{run_size}'][idx] = output["generated_text"]
            
            logger.info(f"Completed distill generation for {run_size}")
            
        except Exception as e:
            logger.error(f"Error processing distill model for {run_size}: {str(e)}")
            
        # Process with vanilla model
        try:
            outputs = vanilla_generators[run_size](
                inputs_dataset["text"],
                batch_size=batch_size,
                num_beams=1,
                do_sample=False,
                max_new_tokens=256  # Add specific max_length
            )
            
            # Update the dataset
            for idx, output in enumerate(outputs):
                if idx < total_samples:
                    samples_dict[f'vanilla_{run_size}'][idx] = output["generated_text"]
            
            logger.info(f"Completed vanilla generation for {run_size}")
            
        except Exception as e:
            logger.error(f"Error processing vanilla model for {run_size}: {str(e)}")
        
        logger.info(f"Completed processing all samples for {run_size}")
        
        # Save intermediate results
        intermediate_df = pd.DataFrame(samples_dict)
        intermediate_df.to_csv(f"{base_path}/generated_answers/intermediate_{run_size}.csv", index=False)
    
    return Dataset.from_dict(samples_dict)


if __name__ == "__main__":
    try:
        # Load models and pipelines for each run size
        vanilla_generators = {}
        distill_generators = {}

        for run_size in run_sizes:
            logger.info(f"Loading models for {run_size}")
            
            # Load vanilla model and tokenizer for this run size
            vanilla_output_path = get_latest_checkpoint(f"{vanilla_base_path}/{run_size}")
            vanilla_tokenizer = AutoTokenizer.from_pretrained(vanilla_output_path)
            vanilla_model = LongT5ForConditionalGeneration.from_pretrained(vanilla_output_path).to(f"cuda:{device}" if device >= 0 else "cpu")

            repetition_penalty = 3.0 if run_size in ['run_2', 'run_5', 'run_7', 'run_10'] else 1.0
            
            # Create a pipeline for vanilla model
            vanilla_generators[run_size] = pipeline(
                "text2text-generation", 
                model=vanilla_model, 
                tokenizer=vanilla_tokenizer, 
                max_new_tokens=256,
                repetition_penalty=repetition_penalty,
                device=device,
            )
            
            # Load distill model and tokenizer for this run size
            distill_output_path = get_latest_checkpoint(f"{distill_base_path}/{run_size}")
            distill_tokenizer = AutoTokenizer.from_pretrained(distill_output_path)
            distill_model = LongT5ForConditionalGeneration.from_pretrained(distill_output_path).to(f"cuda:{device}" if device >= 0 else "cpu")
            
            # Create a pipeline for distill model
            distill_generators[run_size] = pipeline(
                "text2text-generation", 
                model=distill_model, 
                tokenizer=distill_tokenizer, 
                max_new_tokens=256,
                repetition_penalty=repetition_penalty,
                device=device,
            )

        # Process the dataset in batches
        output_dataset = generate_answers_batch(dataset, distill_generators, vanilla_generators)

        # Convert the output dataset to a pandas DataFrame and save directly
        output_df = output_dataset.to_pandas()
        output_df.to_csv(output_file_path, index=False)
        logger.info(f"Successfully saved results to {output_file_path}")

        # Clean up GPU memory
        for generator in list(vanilla_generators.values()) + list(distill_generators.values()):
            if hasattr(generator, 'model'):
                del generator.model
        torch.cuda.empty_cache()

        logger.info("Generation process completed")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise
