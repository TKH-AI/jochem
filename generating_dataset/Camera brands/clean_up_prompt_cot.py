import os
import emoji
from unidecode import unidecode

# Function to convert emojis to unicode
def convert_emojis_to_unicode(text):
    if text is not None:
        return emoji.demojize(text)
    return text

# Function to convert non-ASCII characters to their closest ASCII equivalents
def convert_to_ascii(text):
    if text is not None:
        return unidecode(text)
    return text

# Function to process and save data for the specified folders
def process_and_save_data(folder_prefix, item_type):
    for i in range(1, 7):
        folder_path = f'/Users/jochem/Desktop/mijn_project2/Rationales/LinkedIn/{folder_prefix} {i}'
        
        # Process LinkedIn files
        content_file = os.path.join(folder_path, f'{item_type}_{i}.txt')
        if os.path.exists(content_file):
            with open(content_file, 'r', encoding='utf-8') as cf:
                content = cf.read().strip()
                
                # Convert emojis to unicode and then to ASCII
                content_unicode = convert_emojis_to_unicode(content)
                content_ascii = convert_to_ascii(content_unicode)
                
                # Save the processed content back to the file
                with open(content_file, 'w', encoding='utf-8') as cf:
                    cf.write(content_ascii)
                    
                print(f'Processed and saved {content_file}')
        
        # Process product datasheet description files
        datasheet_file = os.path.join(folder_path, f'datasheet_description_{i}.txt')
        if os.path.exists(datasheet_file):
            with open(datasheet_file, 'r', encoding='utf-8') as dsf:
                datasheet = dsf.read().strip()
                
                # Convert emojis to unicode and then to ASCII
                datasheet_unicode = convert_emojis_to_unicode(datasheet)
                datasheet_ascii = convert_to_ascii(datasheet_unicode)
                
                # Save the processed content back to the file
                with open(datasheet_file, 'w', encoding='utf-8') as dsf:
                    dsf.write(datasheet_ascii)
                    
                print(f'Processed and saved {datasheet_file}')

# Process and save LinkedIn and product datasheet description data
process_and_save_data('LinkedIn', 'linkedin')
process_and_save_data('Rationale', 'rationale')
