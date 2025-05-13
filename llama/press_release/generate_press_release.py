import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

# Constants
MODEL_NAME = "meta-llama/LLaMa-3.2-1B"
TEST_FILE = '/projects/0/prjs1229/resources_press_release_4/test_dataset.csv'
MAX_NEW_TOKENS = 1024
SUBSETS = ['2', '5', '7', '10', '12', '25', '50', '75', '100']
BATCH_SIZE = 64

def find_checkpoint_dir(base_path):
    """Find the checkpoint directory in the model path"""
    # Look for 'checkpoint-' directory
    checkpoint_dirs = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
    if checkpoint_dirs:
        # If there are checkpoint directories, use the latest one
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        return os.path.join(base_path, latest_checkpoint)
    else:
        # If no checkpoint directory found, use the base path
        return base_path

def get_model_paths():
    distill_paths = {}
    vanilla_paths = {}
    
    for subset in SUBSETS:
        distill_paths[subset] = find_checkpoint_dir(f"/projects/0/prjs1229/results_llama/distill/press_release/fourth_run/run_{subset}")
        vanilla_paths[subset] = find_checkpoint_dir(f"/projects/0/prjs1229/results_llama/vanilla/press_release/fourth_run/run_{subset}")
    
    return distill_paths, vanilla_paths

def setup_model_and_tokenizer():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Initialize quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Initialize base model
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="cuda:0",
        torch_dtype=torch.float16
    )
    
    return model, tokenizer

def load_peft_model(base_model, model_path):
    return PeftModel.from_pretrained(base_model, model_path)

def generate_batch_text(model, tokenizer, prompts, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text for multiple prompts in batches"""
    outputs = []
    
    # Keep the seeds for reproducibility
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)
    torch.backends.cudnn.deterministic = True  # Added for extra determinism
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        try:
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to('cuda:0')
            
            with torch.no_grad():
                batch_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.0,
                    do_sample=False,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                )
                
            decoded_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            outputs.extend(decoded_outputs)
            
        except Exception as e:
            logging.error(f"Error processing batch {i}: {str(e)}")
            outputs.extend(["ERROR"] * len(batch_prompts))
            continue
            
    return outputs

def main():
    logging.info("Starting generation process...")
    
    # Load test data
    logging.info(f"Loading test data from {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    logging.info(f"Loaded {len(test_df)} test samples")
    
    # Initialize base model and tokenizer
    logging.info(f"Initializing base model {MODEL_NAME} and tokenizer")
    base_model, tokenizer = setup_model_and_tokenizer()
    logging.info("Model and tokenizer initialized successfully")
    
    # Get model paths with checkpoints
    DISTILL_PATHS, VANILLA_PATHS = get_model_paths()
    logging.info("Model paths loaded:")
    for subset in SUBSETS:
        logging.info(f"Subset {subset}:")
        logging.info(f"  Distill: {DISTILL_PATHS[subset]}")
        logging.info(f"  Vanilla: {VANILLA_PATHS[subset]}")
    
    results = {
        'input': test_df['input'].tolist(),
        'label': test_df['label'].tolist(),
        'rationale': test_df['rationale'].tolist()
    }
    
    # Prepare all prompts at once
    prompts = [
        f"Generate a press release for the following datasheet description: {input_text} Press Release:"
        for input_text in test_df['input']
    ]
    
    # Generate for each model variant
    for subset in SUBSETS:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing subset {subset}")
        logging.info(f"{'='*50}")
        
        # Distill model
        logging.info(f"Loading distill model from {DISTILL_PATHS[subset]}")
        distill_model = load_peft_model(base_model, DISTILL_PATHS[subset])
        
        logging.info("Generating outputs with distill model")
        distill_outputs = generate_batch_text(distill_model, tokenizer, prompts)
        distill_outputs = [
            output.split("Press Release:", 1)[1].strip() if "Press Release:" in output else output
            for output in distill_outputs
        ]
        results[f'distill_run_{subset}'] = distill_outputs
        logging.info(f"Completed distill model generation for subset {subset}")
        
        # Vanilla model
        logging.info(f"Loading vanilla model from {VANILLA_PATHS[subset]}")
        vanilla_model = load_peft_model(base_model, VANILLA_PATHS[subset])
        
        logging.info("Generating outputs with vanilla model")
        vanilla_outputs = generate_batch_text(vanilla_model, tokenizer, prompts)
        vanilla_outputs = [
            output.split("Press Release:", 1)[1].strip() if "Press Release:" in output else output
            for output in vanilla_outputs
        ]
        results[f'vanilla_run_{subset}'] = vanilla_outputs
        logging.info(f"Completed vanilla model generation for subset {subset}")
        
        # Clear GPU memory
        logging.info("Clearing GPU memory")
        del distill_model
        del vanilla_model
        torch.cuda.empty_cache()
    
    # Save results 
    results_df = pd.DataFrame(results)
    
    output_dir = '/projects/0/prjs1229/results_llama/generated_answers/press_release'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'generation_results_full_fourth_run_deterministic.csv')
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    logging.info("Generation process completed successfully")

if __name__ == "__main__":
    main()