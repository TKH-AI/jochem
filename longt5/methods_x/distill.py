import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainerCallback
from typing import Dict, Any, List
import torch
import shutil
import os
import numpy as np
import warnings
from typing import Tuple, Callable
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from transformers.trainer_utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")

# Paths to preprocessed training data
# Paths to preprocessed training data
train_file_paths = {
    '2': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_2.csv',
    '5': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_5.csv', 
    '7': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_7.csv',
    '10': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_10.csv',
    '12': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_12.csv',
    '25': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_25.csv',
    '50': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_50.csv',
    '75': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_75.csv',
    '100': '/home/jheerebrugh/thesis-code-methods/resources_x_4/distill/train_100.csv',
}

# Load the validation dataset
validation_file_path = '/home/jheerebrugh/thesis-code-methods/resources_x_4/validation_dataset.csv'
validation_df = pd.read_csv(validation_file_path)
print(f"validation dataset loaded from {validation_file_path}, number of rows: {len(validation_df)}")

# Convert the validation DataFrame to a Dataset object
validation_dataset = Dataset.from_pandas(validation_df)

# Preprocess the data
def tokenize_function(examples: Dict[str, List[Any]], tokenizer):
    model_inputs = tokenizer(
        ["predict: " + text for text in examples["input"]],
        max_length=2048, 
        truncation=True,
        padding='max_length',
    )
    expl_model_inputs = tokenizer(
        ["explain: " + text for text in examples["input"]],
        max_length=2048, 
        truncation=True,
        padding='max_length',
    )
    model_inputs["expl_input_ids"] = expl_model_inputs["input_ids"]
    model_inputs["expl_attention_mask"] = expl_model_inputs["attention_mask"]

    label_output_encodings = tokenizer(
        text_target=examples["label"], 
        max_length=256, 
        truncation=True, 
        padding='max_length',
    )
    rationale_output_encodings = tokenizer(
        text_target=examples["rationale"],
        max_length=256,
        truncation=True,
        padding='max_length',
    )
    
    # Convert lists of NumPy arrays to single NumPy arrays
    model_inputs["labels"] = label_output_encodings["input_ids"]
    model_inputs["expl_labels"] = rationale_output_encodings["input_ids"]

    return model_inputs

# Preprocess the validation dataset
def preprocess_validation_dataset(tokenizer):
    return validation_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=["input", "label", "rationale"],
        batched=True,
    ).with_format('torch')

CONFIG_DIR = "/home/jheerebrugh/thesis-code-methods/results_x/distill"

class LossLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.label_losses = []
        self.rationale_losses = []
        self.steps = []  # To store step numbers for frequent loss logging

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:  # Training loss
                self.training_losses.append(logs['loss'])
                self.steps.append(state.global_step)  # Keep track of steps for plotting
            if 'eval_loss' in logs:  # Validation loss
                self.validation_losses.append(logs['eval_loss'])
            # Log the label and rationale losses only when they are available
            if 'label_loss' in logs:
                self.label_losses.append(logs['label_loss'])
            else:
                # If label_loss is not available, append a placeholder (e.g., None or previous value)
                self.label_losses.append(self.label_losses[-1] if self.label_losses else 0.0)

            if 'rationale_loss' in logs:
                self.rationale_losses.append(logs['rationale_loss'])
            else:
                # If rationale_loss is not available, append a placeholder
                self.rationale_losses.append(self.rationale_losses[-1] if self.rationale_losses else 0.0)

    def plot_losses(self, save_dir, run_name, num_epochs, alpha, beta):
        steps = np.array(self.steps)  # X-axis is the step numbers

        # Plot training loss over steps
        plt.figure(figsize=(10, 5))
        plt.plot(steps, self.training_losses, marker='o', label='Training Loss')
        plt.title(f'Training Loss over Steps - {run_name} - α={alpha}, β={beta}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/train_loss_{run_name}.png')
        plt.close()

        # Plot validation loss (every epoch) over epochs
        epochs_val = np.arange(1, len(self.validation_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_val, self.validation_losses, marker='x', label='Validation Loss', color='red')
        plt.title(f'Validation Loss over Epochs - {num_epochs} Epochs - {run_name} - α={alpha}, β={beta}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/validation_loss_{run_name}.png')
        plt.close()

        # Ensure that the lengths of steps and losses are the same
        min_len = min(len(steps), len(self.label_losses), len(self.rationale_losses))
        steps = steps[:min_len]
        label_losses = self.label_losses[:min_len]
        rationale_losses = self.rationale_losses[:min_len]

        # Plot label loss and rationale loss over steps
        plt.figure(figsize=(10, 5))
        plt.plot(steps, label_losses, marker='o', label='Label Loss (without alpha)')
        plt.plot(steps, rationale_losses, marker='x', label='Rationale Loss (without beta)')
        plt.title(f'Label Loss vs Rationale Loss over Steps - {run_name} - α={alpha}, β={beta}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/label_rationale_loss_{run_name}.png')
        plt.close()


class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)

        # Generate features for answers
        ans_features = features_df.loc[
            :, features_df.columns.isin(["labels", "input_ids", "attention_mask"])
        ].to_dict("records")
        ans_features = super().__call__(ans_features, return_tensors)

        # Generate features for explanations
        expl_features = (
            features_df.loc[
                :,
                features_df.columns.isin(
                    ["expl_labels", "expl_input_ids", "expl_attention_mask"]
                ),
            ]
            .rename(
                columns={
                    "expl_labels": "labels",
                    "expl_input_ids": "input_ids",
                    "expl_attention_mask": "attention_mask",
                }
            )
            .to_dict("records")
        )
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            "ans": ans_features,
            "expl": expl_features,
        }

# class TaskPrefixTrainer(Seq2SeqTrainer):
#     def __init__(self, alpha, beta, output_rationale, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.beta = beta
#         self.output_rationale = output_rationale

#     def compute_loss(self, model, inputs, return_outputs=False):
#         if "ans" not in inputs or "expl" not in inputs:
#             raise ValueError("Missing 'ans' or 'expl' in inputs")
        
#         ans_outputs = model(**inputs["ans"])
#         expl_outputs = model(**inputs["expl"])

#         #loss = self.alpha * ans_outputs.loss + (1.0 - self.alpha) * expl_outputs.loss
#         loss = self.alpha * ans_outputs.loss + self.beta * expl_outputs.loss #[1, 0.2]

#         return (
#             (loss, {"ans": ans_outputs, "expl": expl_outputs})
#             if return_outputs
#             else loss
#         )
class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, beta, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.output_rationale = output_rationale

    def compute_loss(self, model, inputs, return_outputs=False):
        if "ans" not in inputs or "expl" not in inputs:
            raise ValueError("Missing 'ans' or 'expl' in inputs")

        ans_outputs = model(**inputs["ans"])
        expl_outputs = model(**inputs["expl"])

        # Compute individual losses without alpha/beta for logging
        label_loss = ans_outputs.loss
        rationale_loss = expl_outputs.loss

        # Compute final loss with alpha/beta
        loss = self.alpha * label_loss + self.beta * rationale_loss

        # Log the individual losses without alpha/beta
        self.log({"label_loss": label_loss.item(), "rationale_loss": rationale_loss.item()})

        return (loss, {"ans": ans_outputs, "expl": expl_outputs}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "ans" not in inputs or "expl" not in inputs:
            raise ValueError("Missing 'ans' or 'expl' in inputs")
        
        ans_outputs = super().prediction_step(
            model, 
            inputs["ans"], 
            prediction_loss_only=False, 
            ignore_keys=ignore_keys,
        )
        if self.output_rationale:
            expl_outputs = super().prediction_step(
                model,
                inputs["expl"],
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )
        else:
            expl_outputs = ans_outputs  # placeholder only

        #loss = self.alpha * ans_outputs[0] + (1 - self.alpha) * expl_outputs[0]
        loss = self.alpha * ans_outputs[0] + self.beta * expl_outputs[0]

        return (
            loss,
            [ans_outputs[1], expl_outputs[1]],
            [ans_outputs[2], expl_outputs[2]],
        )

# Define training function
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
        remove_columns=["input", "label", "rationale"],
        batched=True,
    ).with_format('torch')

    # Preprocess the validation dataset with the new tokenizer
    tokenized_validation_dataset = preprocess_validation_dataset(tokenizer)

    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

    # Create unique output directory for this subset's checkpoints and tokenizer
    run_output_dir = os.path.join(CONFIG_DIR, f"run_{subset_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    tokenizer_save_dir = os.path.join(run_output_dir, f"tokenizer_{subset_name}")
    os.makedirs(tokenizer_save_dir, exist_ok=True)

    RUN_ID = 3
    set_seed(RUN_ID)

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_output_dir,  # Set the output directory for checkpoints
        remove_unused_columns=False,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_steps=1,
        logging_steps=50,
        learning_rate=5e-5, #try 5e-5 or 3e-5
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64, #increase
        gradient_accumulation_steps=1,
        num_train_epochs=50,
        predict_with_generate=True,
        seed=RUN_ID,
        prediction_loss_only=False,
        weight_decay=0.0,
        save_total_limit=1,
        local_rank=-1,
        bf16=True,
        load_best_model_at_end=True,
        generation_max_length=256,
    )

    # Initialize the LossLogger
    loss_logger = LossLogger()

    alpha = 1.0
    beta = 0.3

    trainer_kwargs = {
        "alpha": alpha,
        "beta": beta,
        "output_rationale": False, #Must be always set to False
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": tokenized_validation_dataset,
        "data_collator": data_collator,
        "tokenizer": tokenizer,
        "callbacks": [loss_logger],
    }

    # Train the model
    trainer = TaskPrefixTrainer(**trainer_kwargs)
    trainer.train()

    # After training, plot both training and validation loss, and label vs rationale loss
    # loss_logger.plot_losses('/home/jheerebrugh/thesis-code-methods/plots_x/distill', run_name=f'run_{subset_name}', num_epochs=50, alpha=alpha, beta=beta)

# Train on different percentages of the training set
for subset_name in ['2', '5', '7', '10', '12', '25', '50', '75', '100']:
    train_model_on_subset(subset_name)
