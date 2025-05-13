import os
import pandas as pd

def print_faithfulness_results_summary(directory):
    # Get all CSV files in the directory ending with "_faithfulness_evaluation.csv"
    result_files = [f for f in os.listdir(directory) if f.endswith("_faithfulness_evaluation.csv")]

    if not result_files:
        print("No faithfulness evaluation result files found in the directory.")
        return

    # Iterate over each result file to calculate and print the summary
    for file_name in result_files:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        # Check if 'faithfulness_rating' column is present in the dataframe
        if 'faithfulness_rating' not in df.columns:
            print(f"Skipping file '{file_name}' - 'faithfulness_rating' column not found.")
            continue

        # Calculate average rating
        average_rating = df['faithfulness_rating'].mean()

        # Calculate the count of each rating (1, 2, 3, 4, 5)
        rating_counts = df['faithfulness_rating'].value_counts().sort_index()

        # Print the summary for the dataset
        print(f"\nSummary for dataset: {file_name}")
        print(f"Average Faithfulness Rating: {average_rating:.2f}")
        for rating in range(1, 6):
            count = rating_counts.get(rating, 0)
            print(f"Count of {rating}s: {count}")

if __name__ == "__main__":
    # Specify the directory containing the faithfulness evaluation result files
    results_directory = "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic"
    
    # Print the summary for all generated files
    print_faithfulness_results_summary(results_directory)
