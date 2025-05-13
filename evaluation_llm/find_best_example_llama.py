import os
import pandas as pd
from tabulate import tabulate

# Constants
INPUT_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/rated_examples'
OUTPUT_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/comparison_examples'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_examples_by_criteria(df, media_type, size, distill_better=True, n_examples=10):
    """
    Find examples based on specified criteria
    """
    # Create a copy of the filtered dataframe
    filtered_df = df[(df['media_type'] == media_type) & (df['size'] == size)].copy()
    
    # Calculate rating difference
    filtered_df.loc[:, 'rating_difference'] = filtered_df['distill_rating'] - filtered_df['vanilla_rating']
    
    # Sort based on whether we want distill better or worse
    if distill_better:
        sorted_df = filtered_df.nlargest(n_examples, 'rating_difference')
    else:
        sorted_df = filtered_df.nsmallest(n_examples, 'rating_difference')
        
    return sorted_df

def create_markdown_report(examples_dict, output_file):
    """
    Create a detailed markdown report of the examples
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Comparison Examples Analysis\n\n")
        
        for media_type in ['x', 'linkedin', 'press_release']:
            f.write(f"## {media_type.upper()}\n\n")
            
            # 12% examples where distill outperforms
            f.write("### 12% Size (Distill Outperforming)\n\n")
            small_examples_distill = examples_dict[f"{media_type}_12_distill"]
            for idx, row in small_examples_distill.iterrows():
                f.write(f"#### Example {idx + 1}\n")
                f.write(f"**Rating Difference:** {row['rating_difference']:.2f}\n")
                f.write(f"- Distill Rating: {row['distill_rating']}\n")
                f.write(f"- Vanilla Rating: {row['vanilla_rating']}\n\n")
                
                f.write("**Source Text:**\n")
                f.write(f"```\n{row['source']}\n```\n\n")
                
                f.write("**Distill Output:**\n")
                f.write(f"```\n{row['distill_output']}\n```\n\n")
                
                f.write("**Vanilla Output:**\n")
                f.write(f"```\n{row['vanilla_output']}\n```\n\n")
                
                f.write("---\n\n")

            # 12% examples where vanilla outperforms
            f.write("### 12% Size (Vanilla Outperforming)\n\n")
            small_examples_vanilla = examples_dict[f"{media_type}_12_vanilla"]
            for idx, row in small_examples_vanilla.iterrows():
                f.write(f"#### Example {idx + 1}\n")
                f.write(f"**Rating Difference:** {row['rating_difference']:.2f}\n")
                f.write(f"- Distill Rating: {row['distill_rating']}\n")
                f.write(f"- Vanilla Rating: {row['vanilla_rating']}\n\n")
                
                f.write("**Source Text:**\n")
                f.write(f"```\n{row['source']}\n```\n\n")
                
                f.write("**Distill Output:**\n")
                f.write(f"```\n{row['distill_output']}\n```\n\n")
                
                f.write("**Vanilla Output:**\n")
                f.write(f"```\n{row['vanilla_output']}\n```\n\n")
                
                f.write("---\n\n")
            
            # 100% examples where distill outperforms
            f.write("### 100% Size (Distill Outperforming)\n\n")
            large_examples_distill = examples_dict[f"{media_type}_100_distill"]
            for idx, row in large_examples_distill.iterrows():
                f.write(f"#### Example {idx + 1}\n")
                f.write(f"**Rating Difference:** {row['rating_difference']:.2f}\n")
                f.write(f"- Distill Rating: {row['distill_rating']}\n")
                f.write(f"- Vanilla Rating: {row['vanilla_rating']}\n\n")
                
                f.write("**Source Text:**\n")
                f.write(f"```\n{row['source']}\n```\n\n")
                
                f.write("**Distill Output:**\n")
                f.write(f"```\n{row['distill_output']}\n```\n\n")
                
                f.write("**Vanilla Output:**\n")
                f.write(f"```\n{row['vanilla_output']}\n```\n\n")
                
                f.write("---\n\n")

            # 100% examples where vanilla outperforms
            f.write("### 100% Size (Vanilla Outperforming)\n\n")
            large_examples_vanilla = examples_dict[f"{media_type}_100_vanilla"]
            for idx, row in large_examples_vanilla.iterrows():
                f.write(f"#### Example {idx + 1}\n")
                f.write(f"**Rating Difference:** {row['rating_difference']:.2f}\n")
                f.write(f"- Distill Rating: {row['distill_rating']}\n")
                f.write(f"- Vanilla Rating: {row['vanilla_rating']}\n\n")
                
                f.write("**Source Text:**\n")
                f.write(f"```\n{row['source']}\n```\n\n")
                
                f.write("**Distill Output:**\n")
                f.write(f"```\n{row['distill_output']}\n```\n\n")
                
                f.write("**Vanilla Output:**\n")
                f.write(f"```\n{row['vanilla_output']}\n```\n\n")
                
                f.write("---\n\n")

def main():
    print("Starting comparison analysis...")
    
    input_file = os.path.join(INPUT_DIR, 'rated_examples.csv')
    if not os.path.exists(input_file):
        print(f"Error: Could not find rated examples file at {input_file}")
        return
    
    df = pd.read_csv(input_file)
    examples_dict = {}
    
    # Get examples for each media type
    for media_type in ['x', 'linkedin', 'press_release']:
        print(f"\nProcessing {media_type.upper()}...")
        
        # 12% examples (both distill and vanilla outperforming)
        examples_dict[f"{media_type}_12_distill"] = find_examples_by_criteria(
            df, media_type, 12, distill_better=True, n_examples=10
        )
        examples_dict[f"{media_type}_12_vanilla"] = find_examples_by_criteria(
            df, media_type, 12, distill_better=False, n_examples=10
        )
        print(f"Found {len(examples_dict[f'{media_type}_12_distill'])} examples for 12% size (distill better)")
        print(f"Found {len(examples_dict[f'{media_type}_12_vanilla'])} examples for 12% size (vanilla better)")
        
        # 100% examples (both distill and vanilla outperforming)
        examples_dict[f"{media_type}_100_distill"] = find_examples_by_criteria(
            df, media_type, 100, distill_better=True, n_examples=10
        )
        examples_dict[f"{media_type}_100_vanilla"] = find_examples_by_criteria(
            df, media_type, 100, distill_better=False, n_examples=10
        )
        print(f"Found {len(examples_dict[f'{media_type}_100_distill'])} examples for 100% size (distill better)")
        print(f"Found {len(examples_dict[f'{media_type}_100_vanilla'])} examples for 100% size (vanilla better)")
    
    # Create markdown report
    md_output = os.path.join(OUTPUT_DIR, 'comparison_examples_report.md')
    create_markdown_report(examples_dict, md_output)
    print(f"\nDetailed markdown report saved to: {md_output}")
    
    # Save all examples to CSV
    all_examples = pd.concat(examples_dict.values())
    csv_output = os.path.join(OUTPUT_DIR, 'comparison_examples.csv')
    all_examples.to_csv(csv_output, index=False)
    print(f"Examples saved to CSV: {csv_output}")
    
    # Print summary statistics
    print("\nSummary of rating differences:")
    for media_type in ['x', 'linkedin', 'press_release']:
        print(f"\n{media_type.upper()}:")
        for size in [12, 100]:
            for model in ['distill', 'vanilla']:
                key = f"{media_type}_{size}_{model}"
                diffs = examples_dict[key]['rating_difference']
                print(f"{size}% size ({model}) - Average difference: {diffs.mean():.2f}, Min: {diffs.min():.2f}, Max: {diffs.max():.2f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
