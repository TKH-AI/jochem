import pandas as pd
import re

# Load the dataset
df = pd.read_csv('/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_Press_Release_LLaMa_GOOD15.csv')

# Function to clean the content in the 'LinkedIn' and 'Rationale' columns
def clean_text(text):
    # Remove 'content=' and anything that follows 'response_metadata' or 'id='
    text = re.sub(r'content=', '', text)  # Remove 'content='
    text = re.sub(r'response_metadata=.*', '', text)  # Remove 'response_metadata' and everything after it
    text = re.sub(r'id=.*', '', text)  # Remove 'id=' and everything after it
    
    # Replace \\n\\n with actual two newlines
    text = text.replace('\\n\\n', '\n\n')
    
    # Replace single \\n with a newline
    text = text.replace('\\n', '\n')
    
    # Remove unwanted characters, extra spaces
    text = re.sub(r'‚Äú|‚Äù', '"', text)  # Replace malformed quotes
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    
    return text.strip()

# Create a copy of the original dataset to keep the original intact
cleaned_df = df.copy()

# Apply the cleaning function to the 'LinkedIn' and 'Rationale' columns
cleaned_df['Press_Release'] = cleaned_df['Press_Release'].apply(lambda x: clean_text(x) if pd.notnull(x) else x)
cleaned_df['Rationale'] = cleaned_df['Rationale'].apply(lambda x: clean_text(x) if pd.notnull(x) else x)

# Save the cleaned dataset to a new CSV file
cleaned_df.to_csv('/Users/jochem/Desktop/mijn_project2/Camera brands/final_clean_datasets/final_press_release_clean_llama.csv', index=False)

print("Cleaning process completed. Cleaned data saved as 'final_press_release_clean_llama.csv'.")
