import os
import pandas as pd

def print_results(dataset_info):
    for file_path in dataset_info:
        result_file_path = os.path.join('/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic', file_path)
        if not os.path.exists(result_file_path):
            print(f"Result file {result_file_path} does not exist. Skipping.")
            continue

        # Load the evaluation results
        df = pd.read_csv(result_file_path)

        # Print summary statistics for each criterion
        print(f"\nResults for dataset: {file_path.replace('.csv', '')}")
        for criterion in ['relevance', 'attractiveness', 'fluency', 'style_strength']:
            column_name = f'{criterion}_rating'
            if column_name in df.columns:
                avg_rating = df[column_name].mean()
                print(f"Average {criterion} rating: {avg_rating:.2f}")
            else:
                print(f"Column {column_name} not found in {result_file_path}. Skipping.")

def main():
    # Define your synthetic datasets
    synthetic_datasets_info = [
        'final_linkedin_clean_llama_sampled_criteria_evaluation.csv',
        'final_press_release_clean_llama_sampled_criteria_evaluation.csv',
        'final_x_clean_llama_sampled_criteria_evaluation.csv'
    ]

    # Print the results for each dataset
    print_results(synthetic_datasets_info)

if __name__ == "__main__":
    main()
