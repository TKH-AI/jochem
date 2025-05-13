# import os
# from docx import Document
# from docx.table import Table
# from docx.text.paragraph import Paragraph

# # Define the input and output folder paths
# input_folder = '/Users/jochem/Desktop/Press Releases/Allied Vision'
# output_folder = '/Users/jochem/Desktop/Press Releases/Allied Vision/txt_files'

# # Create the output folder if it doesn't exist
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# def iter_block_items(parent):
#     """
#     Yield each paragraph and table child within parent, in document order.
#     Each returned value is an instance of Paragraph or Table.
#     """
#     from docx.oxml import OxmlElement
#     from docx.oxml.ns import qn

#     for child in parent.element.body:
#         if child.tag.endswith('p'):
#             yield Paragraph(child, parent)
#         elif child.tag.endswith('tbl'):
#             yield Table(child, parent)

# # Function to convert docx to txt, including tables
# def convert_docx_to_txt(docx_file, txt_file):
#     # Load the .docx file
#     doc = Document(docx_file)
    
#     # Write the contents of the .docx file into the .txt file
#     with open(txt_file, 'w', encoding='utf-8') as txt_out:
#         for block in iter_block_items(doc):
#             if isinstance(block, Paragraph):
#                 txt_out.write(block.text + '\n')
#             elif isinstance(block, Table):
#                 for row in block.rows:
#                     # Extract text from each cell in the row
#                     row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
#                     # Join the cell data with tabs to preserve table structure
#                     txt_out.write('\t'.join(row_data) + '\n')
#                 # Add an extra newline after each table for readability
#                 txt_out.write('\n')

# # Loop through all the .docx files in the input folder
# for file_name in os.listdir(input_folder):
#     if file_name.lower().endswith('.docx'):
#         # Full path to the docx file
#         docx_file = os.path.join(input_folder, file_name)
        
#         # Define the output .txt file name and path
#         txt_file_name = os.path.splitext(file_name)[0] + '.txt'
#         txt_file = os.path.join(output_folder, txt_file_name)
        
#         # Convert docx to txt
#         try:
#             convert_docx_to_txt(docx_file, txt_file)
#             print(f"Converted: {file_name}")
#         except Exception as e:
#             print(f"Failed to convert {file_name}: {e}")

# print("Conversion complete!")

# import os
# import fitz  # PyMuPDF

# # Define the input and output folder paths
# input_folder = '/Users/jochem/Desktop/Press Releases/FLIR'
# output_folder = '/Users/jochem/Desktop/Press Releases/FLIR/txt_files'

# # Create the output folder if it doesn't exist
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Function to convert PDF to text using PyMuPDF (fitz)
# def convert_pdf_to_txt(pdf_file, txt_file):
#     # Open the PDF file with fitz (PyMuPDF)
#     with fitz.open(pdf_file) as pdf:
#         with open(txt_file, 'w', encoding='utf-8') as txt_file:
#             # Loop through each page
#             for page_num in range(len(pdf)):
#                 page = pdf.load_page(page_num)
#                 text = page.get_text("text")  # Extract text
#                 txt_file.write(text + '\n')   # Write text to file

# # Loop through all the PDF files in the input folder
# for file_name in os.listdir(input_folder):
#     if file_name.endswith('.pdf'):
#         # Full path to the PDF file
#         pdf_file = os.path.join(input_folder, file_name)
        
#         # Define the output .txt file name and path
#         txt_file_name = os.path.splitext(file_name)[0] + '.txt'
#         txt_file = os.path.join(output_folder, txt_file_name)
        
#         # Convert PDF to txt
#         convert_pdf_to_txt(pdf_file, txt_file)

# print("PDF to TXT conversion complete!")

# import pandas as pd

# # Load the dataset
# file_path = '/Users/jochem/Desktop/Press Releases/dataset_Press_Release_aligned_untilallied.xlsx'

# # Read the Excel file into a DataFrame
# df = pd.read_excel(file_path)

# # Count the number of non-empty rows in the 'press release' column
# non_empty_Press_Release_count = df['Press_Release'].notna().sum()

# print(f"Number of filled rows in 'press release': {non_empty_Press_Release_count}")

import pandas as pd
from unidecode import unidecode

# Function to sanitize the text
def sanitize_text(text):
    if pd.isna(text):
        return text
    # Convert non-ASCII characters to their closest ASCII equivalents
    text = unidecode(text)
    
    # Replace specific known problematic characters with a space or an appropriate substitute
    replacements = {
        '•': '-',
        '\u2022': '-',  # Unicode bullet
        '»': '-',
        '™': '(TM)',
        '®': '(R)',
        '©': '(C)',
        'µ': 'u',
        '–': '-',  # en dash
        '—': '-',  # em dash
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '': '',  # unwanted character
        # Add more replacements as needed
    }

    # Apply the replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# File paths
aligned_csv_path = '/Users/jochem/Desktop/Press Releases/dataset_Press_Release_aligned.csv'
final_csv_path = '/Users/jochem/Desktop/Press Releases/final_press_release_clean_llama.csv'
output_csv_path = '/Users/jochem/Desktop/Press Releases/final_press_release_clean_llama_new.csv'

# Load both CSV files
aligned_df = pd.read_csv(aligned_csv_path)
final_df = pd.read_csv(final_csv_path)

# Check if all datasets contain the same number of rows
if len(aligned_df) != len(final_df):
    print(f"Datasets do not have the same number of rows. Aligned: {len(aligned_df)}, Final: {len(final_df)}")
else:
    print(f"Both datasets contain {len(aligned_df)} rows.")

# Check for duplicates in 'camera_model' in both datasets
aligned_duplicates = aligned_df[aligned_df.duplicated('camera_model', keep=False)]
final_duplicates = final_df[final_df.duplicated('camera_model', keep=False)]

if not aligned_duplicates.empty or not final_duplicates.empty:
    print("Duplicate camera_model entries found in the datasets:")
    print("Aligned Dataset Duplicates:\n", aligned_duplicates)
    print("Final Dataset Duplicates:\n", final_duplicates)

    # Remove duplicate rows based on 'camera_model' (optional: keep the first occurrence)
    aligned_df = aligned_df.drop_duplicates(subset='camera_model', keep='first')
    final_df = final_df.drop_duplicates(subset='camera_model', keep='first')

# Sanitize the 'Press_Release' column in the aligned dataset
aligned_df['Press_Release'] = aligned_df['Press_Release'].apply(sanitize_text)

# Merge only on the 'camera_model' column, replacing the 'Press_Release' in final_df
# Set camera_model as the index for both DataFrames to ensure alignment
final_df = final_df.set_index('camera_model')
aligned_df = aligned_df.set_index('camera_model')

# Replace the press releases in final_df where they exist in aligned_df
final_df.update(aligned_df[['Press_Release']])

# Reset index to get back the original structure
final_df.reset_index(inplace=True)

# Save the updated final dataset to a new CSV file
final_df.to_csv(output_csv_path, index=False)
print(f"New dataset saved to {output_csv_path}")


# import pandas as pd

# # Path to the CSV file
# aligned_csv_path = '/Users/jochem/Desktop/Press Releases/dataset_Press_Release_aligned.csv'

# # Load the CSV file into a pandas DataFrame
# df = pd.read_csv(aligned_csv_path)

# # Filter rows where the 'Press_Release' column is not empty
# filled_Press_Release = df[df['Press_Release'].notna()]

# # Group by 'company_name' and count the number of filled rows
# company_filled_counts = filled_Press_Release.groupby('company_name').size()

# # Print the counts for each company
# print(company_filled_counts)

