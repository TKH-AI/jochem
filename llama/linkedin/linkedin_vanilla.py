import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Dict, Any, List
import torch
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from transformers.trainer_utils import set_seed

warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")

# Paths to preprocessed training data
train_file_paths = {
    '2': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_2.csv',
    '5': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_5.csv', 
    '7': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_7.csv',
    '10': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_10.csv',
    '12': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_12.csv',
    '25': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_25.csv',
    '50': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_50.csv',
    '75': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_75.csv',
    '100': '/projects/0/prjs1229/resources_linkedin_4/vanilla/train_100.csv',
}

# Load the validation dataset
validation_file_path = '/projects/0/prjs1229/resources_linkedin_4/validation_dataset.csv'
validation_df = pd.read_csv(validation_file_path)
print(f"Validation dataset loaded from {validation_file_path}, number of rows: {len(validation_df)}")

# Convert the validation DataFrame to a Dataset object 
validation_dataset = Dataset.from_pandas(validation_df)

def tokenize_function(examples: Dict[str, List[Any]], tokenizer):
    # Format prompts
    prompts = [
        f"Generate a marketing post on the platform LinkedIn for the following datasheet description: {inp} Marketing Post: {marketing_copy}"
        for inp, marketing_copy in zip(examples["input"], examples["label"])
    ]
    
    # Tokenize marketing posts
    model_inputs = tokenizer(
        prompts,
        max_length=2560,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    # Create attention mask
    model_inputs["attention_mask"] = model_inputs["input_ids"].ne(tokenizer.pad_token_id)
    
    # Labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # Mask out the loss for input portions
    for i in range(len(prompts)):
        # Mask input portion
        input_portion = f"Generate a marketing post on the platform LinkedIn for the following datasheet description: {examples['input'][i]} Marketing Post:"
        input_length = len(tokenizer(input_portion)["input_ids"])
        model_inputs["labels"][i, :input_length] = -100
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"]
    }

# Preprocess the validation dataset
def preprocess_validation_dataset(tokenizer):
    return validation_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label"],
        batched=True,
    ).with_format('torch')

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Stack all tensors in the batch
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        return batch

CONFIG_DIR = "/projects/0/prjs1229/results_llama/vanilla/linkedin/fourth_run"

class LossLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.validation_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)

    def plot_losses(self, save_dir, run_name):
        os.makedirs(save_dir, exist_ok=True)
        
        # Combined plot
        plt.figure(figsize=(12, 6))
        if self.training_losses:
            plt.plot(self.steps, self.training_losses, label='Training Loss', alpha=0.6)
        if self.validation_losses:
            plt.plot(self.eval_steps, self.validation_losses, 
                    marker='x', markersize=8, 
                    label='Validation Loss', color='red')
            
        plt.title(f'Training and Validation Loss\n{run_name}')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'losses_{run_name}.png'), dpi=300)
        plt.close()

def get_training_parameters(subset_size):
    """Calculate training parameters based on dataset size"""
    # Map subset sizes to (eval_steps, max_steps, logging_steps)
    parameter_mapping = {
        '2': (10, 300, 5),     # Smallest dataset: log very frequently
        '5': (15, 400, 5),
        '7': (20, 500, 8),
        '10': (30, 700, 10),
        '12': (40, 800, 10),
        '25': (50, 1000, 15),
        '50': (60, 1500, 20),  # Larger dataset: log less frequently
        '75': (80, 1800, 20),  # Larger dataset: log less frequently
        '100': (80, 1800, 20),  # Larger dataset: log less frequently

    }
    
    return parameter_mapping[subset_size]

def train_model_on_subset(subset_name):
    # Load datasets
    train_file_path = train_file_paths[subset_name]
    train_df = pd.read_csv(train_file_path)
    train_dataset = Dataset.from_pandas(train_df)
    
    # Setup model and tokenizer
    model_name = "meta-llama/LLaMa-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",  # Explicitly specify the GPU device
        torch_dtype=torch.float16
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Update LoRA config for 4-bit training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Add k_proj and o_proj
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]  # Add this line
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess datasets
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label"],
        batched=True,
    ).with_format('torch')
    
    tokenized_validation_dataset = preprocess_validation_dataset(tokenizer)
    
    # Setup training
    run_output_dir = os.path.join(CONFIG_DIR, f"run_{subset_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    RUN_ID = 3
    set_seed(RUN_ID)

    data_collator = DataCollator(tokenizer=tokenizer)

    eval_steps, max_steps, logging_steps = get_training_parameters(subset_name)
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        remove_unused_columns=False,
        eval_strategy='steps',  # Change to steps-based evaluation
        eval_steps=eval_steps,  # Dynamic evaluation steps
        save_strategy='steps',
        save_steps=eval_steps,  # Match save steps to eval steps
        logging_steps=logging_steps,
        lr_scheduler_type="cosine",  # Change from default linear
        learning_rate=5e-5,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        seed=RUN_ID,
        weight_decay=0.01,
        save_total_limit=1,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Initialize the LossLogger
    loss_logger = LossLogger()

    # Initialize callbacks including early stopping
    callbacks = [
        loss_logger,
        EarlyStoppingCallback(
            early_stopping_patience=5,        # Increased from 3 to 5
            early_stopping_threshold=0.005,   # Decreased from 0.01 to 0.005
        )
    ]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    trainer.train()
    
    loss_logger.plot_losses(
        "/projects/0/prjs1229/plots_llama/vanilla/linkedin/fourth_run",
        run_name=f'run_{subset_name}'
    )

if __name__ == "__main__":
    for subset_name in ['2', '5', '7', '10', '12', '25', '50', '75', '100']:
        train_model_on_subset(subset_name)