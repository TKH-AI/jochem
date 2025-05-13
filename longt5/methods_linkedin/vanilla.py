import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainerCallback
from typing import Dict, Any, List
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers.trainer_utils import set_seed
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")

# Paths to preprocessed training data
train_file_paths = {
    '2': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_2.csv',
    '5': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_5.csv', 
    '7': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_7.csv',
    '10': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_10.csv',
    '12': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_12.csv',
    '25': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_25.csv',
    '50': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_50.csv',
    '75': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_75.csv',
    '100': '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/vanilla/train_100.csv',
}

# Load the validation dataset
validation_file_path = '/home/jheerebrugh/thesis-code-methods/resources_linkedin_4/validation_dataset.csv'
validation_df = pd.read_csv(validation_file_path)
print(f"validation dataset loaded from {validation_file_path}, number of rows: {len(validation_df)}")

# Convert the validation DataFrame to a Dataset object
validation_dataset = Dataset.from_pandas(validation_df)

# Preprocess the data
def tokenize_function(examples: Dict[str, List[Any]], tokenizer):
    inputs = ["predict: " + text for text in examples["input"]]

    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding='max_length',
        truncation=True,
    )

    labels = tokenizer(
        text_target=[text if text is not None else "" for text in examples["label"]],
        max_length=512,
        padding='max_length',
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Preprocess the validation dataset
def preprocess_validation_dataset(tokenizer):
    return validation_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label"],
        batched=True,
    ).with_format('torch')

CONFIG_DIR = "/home/jheerebrugh/thesis-code-methods/results_linkedin/vanilla"

class LossLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:  # Training loss
                self.training_losses.append(logs['loss'])
            if 'eval_loss' in logs:  # Validation loss
                self.validation_losses.append(logs['eval_loss'])

    def plot_losses(self, save_dir, run_name, num_epochs):
        epochs_train = np.arange(1, len(self.training_losses) + 1)

        # Plot training loss separately
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_train, self.training_losses, marker='o', label='Training Loss')
        plt.title(f'Training Loss over Epochs - {num_epochs} Epochs - {run_name}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/train_loss_{run_name}.png')
        plt.close()

        # Plot validation loss separately
        epochs_val = np.arange(1, len(self.validation_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_val, self.validation_losses, marker='x', label='Validation Loss', color='red')
        plt.title(f'Validation Loss over Epochs - {num_epochs} Epochs - {run_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/validation_loss_{run_name}.png')
        plt.close()

# Define training arguments
def train_model_on_subset(subset_name):
    # Load the training subset
    train_file_path = train_file_paths[subset_name]
    train_df = pd.read_csv(train_file_path)
    print(f"Train subset {subset_name} loaded from {train_file_path}, number of rows: {len(train_df)}")

    # Convert the training DataFrame to a Dataset object
    train_dataset = Dataset.from_pandas(train_df)

    # Initialize the tokenizer and model for each training subset
    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base")

    # Preprocess the training dataset
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label"],
        batched=True,
    ).with_format('torch')

    # Preprocess the validation dataset with the new tokenizer
    tokenized_validation_dataset = preprocess_validation_dataset(tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Create unique output directory for this subset's checkpoints and tokenizer
    run_output_dir = os.path.join(CONFIG_DIR, f"run_{subset_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    tokenizer_save_dir = os.path.join(run_output_dir, f"tokenizer_{subset_name}")
    os.makedirs(tokenizer_save_dir, exist_ok=True)

    RUN_ID = 3
    set_seed(RUN_ID)

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_output_dir,  # Set the output directory for checkpoints
        eval_strategy='epoch',
        save_strategy='epoch',
        save_steps=1,
        logging_steps=50,
        learning_rate=5e-5, #try 5e-5 or 3e-5
        per_device_train_batch_size=8, #5
        per_device_eval_batch_size=64, #5
        gradient_accumulation_steps=1,
        num_train_epochs=50, #3
        predict_with_generate=True,
        seed=RUN_ID,
        prediction_loss_only=False,
        weight_decay=0.0,
        save_total_limit=1,
        local_rank=-1,
        bf16=True,
        load_best_model_at_end=True,
        generation_max_length=512,
    )

    # Initialize the LossLogger
    loss_logger = LossLogger()
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": tokenized_validation_dataset,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
        "callbacks": [loss_logger],
    }

    # Train the model
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()

    # After training, plot the loss
    # loss_logger.plot_losses('/home/jheerebrugh/thesis-code-methods/plots_linkedin/vanilla_3', run_name=f'run_{subset_name}', num_epochs=50)

# Train on different percentages of the training set
for subset_name in ['2', '5', '7', '10', '12', '25', '50', '75', '100']:
    train_model_on_subset(subset_name)
