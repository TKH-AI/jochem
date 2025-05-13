import os
import pandas as pd
from datasets import Dataset

# Load the CSV file into a pandas DataFrame
file_path = '/Users/jochem/Desktop/thesis-code-methods/resources_x/final_x_clean_llama.csv'
df = pd.read_csv(file_path)

# Vanilla dataset preparation
df_vanilla = df[['datasheet_description', 'X']]
df_vanilla.columns = ['input', 'label']
df_vanilla_filtered = df_vanilla.dropna(subset=['input', 'label'])
df_vanilla_filtered.reset_index(drop=True, inplace=True)

# Distill dataset preparation
df_distill = df[['datasheet_description', 'X', 'Rationale']]
df_distill.columns = ['input', 'label', 'rationale']
df_distill_filtered = df_distill.dropna(subset=['input', 'label', 'rationale'])
df_distill_filtered.reset_index(drop=True, inplace=True)

# Combine both filtered datasets to maintain consistent splits
combined_df = pd.merge(df_vanilla_filtered, df_distill_filtered, on=['input', 'label'], how='inner')

# Convert the combined DataFrame to a Dataset object
dataset_combined = Dataset.from_pandas(combined_df)

# Split the combined dataset into training, validation, and test sets
train_valid_split = dataset_combined.train_test_split(test_size=0.2, seed=45)
valid_test_split = train_valid_split['test'].train_test_split(test_size=0.5, seed=45)

combined_splits = {
    'train': train_valid_split['train'],
    'validation': valid_test_split['train'],
    'test': valid_test_split['test']
}

# Remove 'rationale' column for the vanilla dataset splits
dataset_vanilla_split = {
    'train': combined_splits['train'].remove_columns(['rationale']),
    'validation': combined_splits['validation'].remove_columns(['rationale']),
    'test': combined_splits['test'].remove_columns(['rationale'])
}

# Create folders to store the datasets
output_dir = '/Users/jochem/Desktop/thesis-code-methods/resources_x_4/'
os.makedirs(output_dir, exist_ok=True)
vanilla_folder = os.path.join(output_dir, 'vanilla')
distill_folder = os.path.join(output_dir, 'distill')
os.makedirs(vanilla_folder, exist_ok=True)
os.makedirs(distill_folder, exist_ok=True)

# Save the test dataset (shared between vanilla and distill)
test_file_path = os.path.join(output_dir, 'test_dataset.csv')
combined_splits['test'].to_pandas().to_csv(test_file_path, index=False)
print(f"Test dataset saved to {test_file_path}, number of rows: {len(combined_splits['test'])}")

# Define a function to save subsets of training data
def save_train_subset(dataset, folder_path, subset_name, subset_percentage):
    subset_train_dataset = dataset['train'].shuffle(seed=45).select(range(int(len(dataset['train']) * subset_percentage)))
    subset_file_path = os.path.join(folder_path, f"train_{subset_name}.csv")
    subset_train_dataset.to_pandas().to_csv(subset_file_path, index=False)
    subset_df = pd.read_csv(subset_file_path)
    print(f"Train subset {subset_name} saved to {subset_file_path}, number of rows: {len(subset_df)}")

# Save training subsets for both vanilla and distill datasets
for subset_percentage in [0.025, 0.05, 0.075, 0.1, 0.125, 0.25, 0.5, 0.75, 1.0]:
    subset_name = f'{int(subset_percentage * 100)}'
    save_train_subset(dataset_vanilla_split, vanilla_folder, subset_name, subset_percentage)
    save_train_subset(combined_splits, distill_folder, subset_name, subset_percentage)

# Save the validation dataset (shared between vanilla and distill)
validation_file_path = os.path.join(output_dir, 'validation_dataset.csv')
combined_splits['validation'].to_pandas().to_csv(validation_file_path, index=False)
print(f"Validation dataset saved to {validation_file_path}, number of rows: {len(combined_splits['validation'])}")
