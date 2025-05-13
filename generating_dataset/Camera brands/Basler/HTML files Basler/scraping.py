import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# The base URL of the website to scrape
base_url = 'https://en.ids-imaging.com/downloads.html'

# Directory to save downloaded PDFs
save_dir = '/Users/jochem/Desktop/HTML files Basler/downlaoded_pdfs'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

visited_urls = set()

# Create a requests session
session = requests.Session()
session.max_redirects = 5  

def scrape_pdfs(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links ending with .pdf and containing "Datasheet"
        pdf_links = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))
        for link in pdf_links:
            pdf_url = urljoin(url, link['href'])
            if pdf_url not in visited_urls:
                download_pdf(pdf_url)
                visited_urls.add(pdf_url)

        # Find all other links to follow
        page_links = soup.find_all('a', href=True)
        for link in page_links:
            full_url = urljoin(url, link['href'])
            if is_valid_url(full_url) and full_url not in visited_urls:
                visited_urls.add(full_url)
                scrape_pdfs(full_url)
                
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape {url}: {e}")

def download_pdf(pdf_url):
    try:
        pdf_response = session.get(pdf_url)
        pdf_response.raise_for_status()  # Raise an error for bad status codes
        pdf_name = os.path.join(save_dir, os.path.basename(pdf_url))
        
        with open(pdf_name, 'wb') as pdf_file:
            pdf_file.write(pdf_response.content)
        
        print(f'Downloaded: {pdf_name}')
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdf_url}: {e}")

def is_valid_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme in ('http', 'https') and parsed_url.netloc.endswith('en.ids-imaging.com')

# Start scraping from the base URL
scrape_pdfs(base_url)

print('All PDFs with "Datasheet" in their name have been downloaded.')
