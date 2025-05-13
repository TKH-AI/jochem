import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM,  # Changed from LongT5ForConditionalGeneration
    TrainingArguments,  # Changed from Seq2SeqTrainingArguments
    Trainer,  # Changed from Seq2SeqTrainer
    TrainerCallback,
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
import shutil
import os
import numpy as np
import warnings
from typing import Tuple, Callable
import matplotlib.pyplot as plt
from transformers.trainer_utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")

# Paths to preprocessed training data
train_file_paths = {
    '2': '/projects/0/prjs1229/resources_x_4/distill/train_2.csv',
    '5': '/projects/0/prjs1229/resources_x_4/distill/train_5.csv',
    '7': '/projects/0/prjs1229/resources_x_4/distill/train_7.csv',
    '10': '/projects/0/prjs1229/resources_x_4/distill/train_10.csv',
    '12': '/projects/0/prjs1229/resources_x_4/distill/train_12.csv',
    '25': '/projects/0/prjs1229/resources_x_4/distill/train_25.csv',
    '50': '/projects/0/prjs1229/resources_x_4/distill/train_50.csv',
    '75': '/projects/0/prjs1229/resources_x_4/distill/train_75.csv',
    '100': '/projects/0/prjs1229/resources_x_4/distill/train_100.csv',
}

# Load the validation dataset
validation_file_path = '/projects/0/prjs1229/resources_x_4/validation_dataset.csv'
validation_df = pd.read_csv(validation_file_path)
print(f"validation dataset loaded from {validation_file_path}, number of rows: {len(validation_df)}")

# Convert the validation DataFrame to a Dataset object
validation_dataset = Dataset.from_pandas(validation_df)

def tokenize_function(examples: Dict[str, List[Any]], tokenizer):
    # Format prompts for both marketing posts and rationales
    marketing_prompts = [
        f"Generate a marketing post on the platform X for the following datasheet description: {inp} Marketing Post: {marketing_copy}" 
        for inp, marketing_copy in zip(examples["input"], examples["label"])
    ]
    
    rationale_prompts = [
        f"Generate a rationale behind writing a marketing post on the platform X for the following datasheet description: {inp} Rationale: {rationale}"
        for inp, rationale in zip(examples["input"], examples["rationale"])
    ]
    
    # Tokenize marketing posts
    marketing_inputs = tokenizer(
        marketing_prompts,
        max_length=2304,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize rationales
    rationale_inputs = tokenizer(
        rationale_prompts,
        max_length=2304,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    # Create attention masks
    marketing_inputs["attention_mask"] = marketing_inputs["input_ids"].ne(tokenizer.pad_token_id)
    rationale_inputs["attention_mask"] = rationale_inputs["input_ids"].ne(tokenizer.pad_token_id)
    
    # Labels are the same as input_ids
    marketing_inputs["labels"] = marketing_inputs["input_ids"].clone()
    rationale_inputs["labels"] = rationale_inputs["input_ids"].clone()
    
    # Mask out the loss for input portions
    for i in range(len(marketing_prompts)):
        # Mask marketing post input
        marketing_input = f"Generate a marketing post on the platform X for the following datasheet description: {examples['input'][i]} Marketing Post:"
        marketing_length = len(tokenizer(marketing_input)["input_ids"])
        marketing_inputs["labels"][i, :marketing_length] = -100
        
        # Mask rationale input
        rationale_input = f"Generate a rationale behind writing a marketing post on the platform X for the following datasheet description: {examples['input'][i]} Rationale:"
        rationale_length = len(tokenizer(rationale_input)["input_ids"])
        rationale_inputs["labels"][i, :rationale_length] = -100
    
    return {
        "marketing_input_ids": marketing_inputs["input_ids"],
        "marketing_attention_mask": marketing_inputs["attention_mask"],
        "marketing_labels": marketing_inputs["labels"],
        "rationale_input_ids": rationale_inputs["input_ids"],
        "rationale_attention_mask": rationale_inputs["attention_mask"],
        "rationale_labels": rationale_inputs["labels"]
    }

# Preprocess the validation dataset
def preprocess_validation_dataset(tokenizer):
    return validation_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label", "rationale"],
        batched=True,
    ).with_format('torch')

CONFIG_DIR = "/projects/0/prjs1229/results_llama/distill/x/fourth_run"

class LossLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.label_losses = []
        self.rationale_losses = []
        self.eval_label_losses = []
        self.eval_rationale_losses = []
        self.steps = []
        self.eval_steps = []  # Add this to track evaluation steps separately

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Training metrics
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'label_loss' in logs and not any(k.startswith('eval_') for k in logs):
                self.label_losses.append(logs['label_loss'])
            if 'rationale_loss' in logs and not any(k.startswith('eval_') for k in logs):
                self.rationale_losses.append(logs['rationale_loss'])
                
            # Validation metrics
            if 'eval_loss' in logs:
                self.validation_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)  # Store the step at which evaluation occurred
            if 'eval_label_loss' in logs:
                self.eval_label_losses.append(logs['eval_label_loss'])
            if 'eval_rationale_loss' in logs:
                self.eval_rationale_losses.append(logs['eval_rationale_loss'])

    def plot_losses(self, save_dir, run_name, alpha, beta):
        os.makedirs(save_dir, exist_ok=True)
        steps = np.array(self.steps)
        
        # Plot training and validation combined losses
        plt.figure(figsize=(12, 6))
        if self.training_losses:
            plt.plot(steps, self.training_losses, label='Training Loss', alpha=0.6)
        
        # Plot validation loss points
        if self.validation_losses and self.eval_steps:
            plt.plot(self.eval_steps, self.validation_losses, 
                    marker='x', markersize=8, 
                    label='Validation Loss', color='red')
        
        plt.title(f'Training and Validation Loss\n{run_name} (α={alpha}, β={beta})')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'train_val_loss_{run_name}.png'), dpi=300)
        plt.close()

        # Plot component losses
        plt.figure(figsize=(12, 6))
        
        # Plot training component losses
        if self.label_losses and self.rationale_losses:
            min_len = min(len(steps), len(self.label_losses), len(self.rationale_losses))
            steps_truncated = steps[:min_len]
            plt.plot(steps_truncated, self.label_losses[:min_len], 
                    alpha=0.6, label='Train Label Loss')
            plt.plot(steps_truncated, self.rationale_losses[:min_len], 
                    alpha=0.6, label='Train Rationale Loss')
        
        # Plot validation component losses
        if self.eval_label_losses and self.eval_rationale_losses and self.eval_steps:
            plt.plot(self.eval_steps[:len(self.eval_label_losses)], 
                    self.eval_label_losses, marker='x', 
                    markersize=8, label='Val Label Loss', linestyle='--')
            plt.plot(self.eval_steps[:len(self.eval_rationale_losses)], 
                    self.eval_rationale_losses, marker='x', 
                    markersize=8, label='Val Rationale Loss', linestyle='--')
        
        plt.title(f'Component Losses\n{run_name} (α={alpha}, β={beta})')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'component_losses_{run_name}.png'), dpi=300)
        plt.close()

class TaskPrefixDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Convert list of dicts to dict of lists and stack them
        batch = {
            "ans": {
                "input_ids": torch.stack([f["marketing_input_ids"] for f in features]),
                "attention_mask": torch.stack([f["marketing_attention_mask"] for f in features]),
                "labels": torch.stack([f["marketing_labels"] for f in features])
            },
            "expl": {
                "input_ids": torch.stack([f["rationale_input_ids"] for f in features]),
                "attention_mask": torch.stack([f["rationale_attention_mask"] for f in features]),
                "labels": torch.stack([f["rationale_labels"] for f in features])
            }
        }
        return batch

class TaskPrefixTrainer(Trainer):
    def __init__(self, alpha, beta, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.output_rationale = output_rationale
        self.tokenizer = kwargs.get('tokenizer', None)
        # Initialize loss tracking attributes
        self._label_loss = None
        self._rationale_loss = None
        self._combined_loss = None

    def compute_loss(self, model, inputs, return_outputs=False):
        if "ans" not in inputs or "expl" not in inputs:
            raise ValueError("Missing 'ans' or 'expl' in inputs")

        # Forward pass for marketing posts and rationales
        marketing_outputs = model(**inputs["ans"])
        rationale_outputs = model(**inputs["expl"])

        # Get individual losses and ensure they're detached
        self._label_loss = marketing_outputs.loss.detach()
        self._rationale_loss = rationale_outputs.loss.detach()

        # Combine losses using alpha and beta weights
        loss = self.alpha * marketing_outputs.loss + self.beta * rationale_outputs.loss
        
        # Store combined loss
        self._combined_loss = loss.detach()

        if return_outputs:
            return loss, {"ans": marketing_outputs, "expl": rationale_outputs}
        return loss

    def log(self, logs):
        # Always add the component losses to logs
        logs = logs.copy()  # Create a copy to avoid modifying the original
        
        # Determine if we're in evaluation mode
        is_eval = any(k.startswith("eval_") for k in logs)
        prefix = "eval_" if is_eval else ""
        
        # Add component losses if they exist
        if self._label_loss is not None:
            logs[f"{prefix}label_loss"] = self._label_loss.item()
        if self._rationale_loss is not None:
            logs[f"{prefix}rationale_loss"] = self._rationale_loss.item()
        if self._combined_loss is not None and is_eval:
            logs["eval_loss"] = self._combined_loss.item()

        # Call parent's log method
        super().log(logs)
        
        # Reset loss tracking attributes after logging
        if not is_eval:
            self._label_loss = None
            self._rationale_loss = None
            self._combined_loss = None

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            with torch.no_grad():
                # Compute losses for both tasks
                marketing_outputs = model(**inputs["ans"])
                rationale_outputs = model(**inputs["expl"])
                
                # Store component losses
                self._label_loss = marketing_outputs.loss.detach()
                self._rationale_loss = rationale_outputs.loss.detach()
                
                # Compute combined loss
                loss = self.alpha * self._label_loss + self.beta * self._rationale_loss
                self._combined_loss = loss.detach()
                
                return (loss, None, None)

        # For generation
        with torch.no_grad():
            marketing_outputs = model.generate(
                input_ids=inputs["ans"]["input_ids"],
                attention_mask=inputs["ans"]["attention_mask"],
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            if self.output_rationale:
                rationale_outputs = model.generate(
                    input_ids=inputs["expl"]["input_ids"],
                    attention_mask=inputs["expl"]["attention_mask"],
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                rationale_outputs = None

        return None, [marketing_outputs, rationale_outputs], None

def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for decoder models
    tokenizer.padding_side = 'left'
    return tokenizer

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

# Define training function
def train_model_on_subset(subset_name):
    # Load the training subset
    train_file_path = train_file_paths[subset_name]
    train_df = pd.read_csv(train_file_path)
    print(f"Train subset {subset_name} loaded from {train_file_path}, number of rows: {len(train_df)}")

    # Convert the training DataFrame to a Dataset object
    train_dataset = Dataset.from_pandas(train_df)

    # Initialize the tokenizer and model
    model_name = "meta-llama/LLaMa-3.2-1B"
    tokenizer = setup_tokenizer(model_name)
    
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

    # Preprocess the training dataset
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label", "rationale"],
        batched=True,
    ).with_format('torch')

    # Preprocess the validation dataset
    tokenized_validation_dataset = preprocess_validation_dataset(tokenizer)

    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer)

    # Create unique output directory for this subset's checkpoints and tokenizer
    run_output_dir = os.path.join(CONFIG_DIR, f"run_{subset_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    RUN_ID = 3
    set_seed(RUN_ID)

    eval_steps, max_steps, logging_steps = get_training_parameters(subset_name)
    # Training arguments
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
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
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

    alpha = 1.0
    beta = 0.3

    trainer_kwargs = {
        "alpha": alpha,
        "beta": beta,
        "output_rationale": False,
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": tokenized_validation_dataset,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
        "callbacks": callbacks,  # Updated callbacks list
    }

    # Train the model
    trainer = TaskPrefixTrainer(**trainer_kwargs)
    trainer.train()

    # In your train_model_on_subset function, update the plot_losses call:
    loss_logger.plot_losses(
        "/projects/0/prjs1229/plots_llama/distill/x/fourth_run",
        run_name=f'run_{subset_name}',
        alpha=alpha,
        beta=beta
    )

# Main execution
if __name__ == "__main__":
    # Train on different percentages of the training set
    for subset_name in ['2', '5', '7', '10', '12', '25', '50', '75', '100']:
        train_model_on_subset(subset_name)