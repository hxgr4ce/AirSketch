import os
import requests
from concurrent.futures import ThreadPoolExecutor
import json
import argparse

def download_file(url, directory):
    """Download a file from a URL into the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    local_filename = url.split('/')[-1].replace('%20', ' ')
    full_path = os.path.join(directory, local_filename)

    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {local_filename} to {directory}")

def main(categories, base_url, directory):
    """Download multiple categories in parallel."""
    urls = [f"{base_url}/{category.replace(' ', '%20')}.ndjson" for category in categories]
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(download_file, urls, [directory] * len(urls))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, default="quickdraw_categories.json", help="path to json file that lists categories")
    parser.add_argument("--output_directory", type=str, default="datasets/quickdraw_ndjsons_raw_all", help="path to directory where ndjsons are saved")

    args = parser.parse_args()

    categories = json.load(open(args.categories, 'r'))
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/raw"
    directory = args.output_directory

    main(categories, base_url, directory)
