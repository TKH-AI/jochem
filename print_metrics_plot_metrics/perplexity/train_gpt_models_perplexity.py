import logging
from datetime import datetime
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
import os
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gpt2_finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class GPT2FineTuner:
    def __init__(self, model_name='gpt2-medium', max_length=1024):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
    def prepare_data(self, train_path, val_path):
        """Load and prepare data for training"""
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            
            # Train only on marketing copy (label) for style evaluation
            def combine_text(row):
                # Only use the label (marketing copy) for training
                return row['label']
            
            logging.info("Training model on marketing copy (label) only for style evaluation")
            train_texts = train_df.apply(combine_text, axis=1).tolist()
            val_texts = val_df.apply(combine_text, axis=1).tolist()
            
            # Filter out any empty texts
            train_texts = [text for text in train_texts if isinstance(text, str) and text.strip()]
            val_texts = [text for text in val_texts if isinstance(text, str) and text.strip()]
            
            return train_texts, val_texts
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise
            
    def tokenize_data(self, texts, tokenizer):
        """Tokenize texts for model input"""
        try:
            encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Create dataset with labels equal to input_ids for language modeling
            dataset = Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': encodings['input_ids'].clone()  # Add labels for loss calculation
            })
            
            return dataset
            
        except Exception as e:
            logging.error(f"Error tokenizing data: {str(e)}")
            raise

            
    def fine_tune(self, dataset_name, train_path, val_path, output_dir):
        """Fine-tune GPT-2 on specific dataset"""
        try:
            logging.info(f"Starting fine-tuning for {dataset_name}")
            
            # Initialize model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Add padding token
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
            
            # Prepare data
            train_texts, val_texts = self.prepare_data(train_path, val_path)
            train_dataset = self.tokenize_data(train_texts, tokenizer)
            val_dataset = self.tokenize_data(val_texts, tokenizer)
            
            # Define training arguments with early stopping
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=10,  # Maximum number of epochs
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=5e-5,
                lr_scheduler_type="linear",
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f'{output_dir}/logs',
                logging_steps=10,
                save_strategy='epoch',
                evaluation_strategy="epoch",  # Evaluate after each epoch
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",  # Use validation loss as metric
                greater_is_better=False,  # Lower loss is better
                gradient_accumulation_steps=4,
                fp16=torch.cuda.is_available()
            )
            
            # Initialize trainer with early stopping
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=3,  # Stop if no improvement for 3 epochs
                        early_stopping_threshold=0.01  # Minimum change to qualify as an improvement
                    )
                ]
            )
            
            # Train model
            trainer.train()
            
            # Log final metrics
            eval_results = trainer.evaluate()
            logging.info(f"Final evaluation metrics for {dataset_name}:")
            logging.info(f"Validation Loss: {eval_results['eval_loss']:.4f}")
            
            # Save final model and tokenizer
            final_output_dir = f"{output_dir}/final"
            os.makedirs(final_output_dir, exist_ok=True)
            trainer.save_model(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            
            logging.info(f"Completed fine-tuning for {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error during fine-tuning {dataset_name}: {str(e)}")
            raise
def main():
    # Define dataset paths for each run
    base_datasets = {
        'linkedin': {
            'base_train': '/projects/0/prjs1229/resources_linkedin_{}/vanilla/train_100.csv',
            'base_val': '/projects/0/prjs1229/resources_linkedin_{}/validation_dataset.csv',
            'base_output': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_linkedin_run{}'
        },
        'press_release': {
            'base_train': '/projects/0/prjs1229/resources_press_release_{}/vanilla/train_100.csv',
            'base_val': '/projects/0/prjs1229/resources_press_release_{}/validation_dataset.csv',
            'base_output': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_press_release_run{}'
        },
        'x': {
            'base_train': '/projects/0/prjs1229/resources_x_{}/vanilla/train_100.csv',
            'base_val': '/projects/0/prjs1229/resources_x_{}/validation_dataset.csv',
            'base_output': '/home/jheerebrugh/thesis-code-methods/gpt_models/gpt2_x_run{}'
        }
    }
    
        # Initialize fine-tuner
    fine_tuner = GPT2FineTuner()
    
    # Train models for each run (1-4)
    for run in range(1, 5):
        logging.info(f"Starting training for run {run}")
        
        # Create datasets dictionary for current run
        datasets = {}
        for dataset_name, paths in base_datasets.items():
            datasets[dataset_name] = {
                'train': paths['base_train'].format(run),
                'val': paths['base_val'].format(run),
                'output': paths['base_output'].format(run)
            }
        
        # Fine-tune for each dataset in current run
        for dataset_name, paths in datasets.items():
            try:
                logging.info(f"Starting processing for {dataset_name} - Run {run}")
                os.makedirs(paths['output'], exist_ok=True)
                
                fine_tuner.fine_tune(
                    f"{dataset_name}_run{run}",
                    paths['train'],
                    paths['val'],
                    paths['output']
                )
                
            except Exception as e:
                logging.error(f"Error processing {dataset_name} - Run {run}: {str(e)}")
                continue

if __name__ == "__main__":
    main()