import pandas as pd
import requests
import gzip
import shutil
import os
import re
import uuid
import hashlib

# ---------- CONFIGURATION ----------
BASE_URL = "https://dumps.wikimedia.org/other/pageviews"
YEAR = 2024
MONTH = 5
DAY = 1
HOURS = range(0, 1)  
OUTPUT_CSV = "data.csv"
CHUNK_SIZE = 100000  

# ---------- HELPERS ----------
def download_file(url, local_path):
    print(f"⬇ Downloading {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print(f"✅ Downloaded to {local_path}")
    else:
        print(f"❌ Failed to download {url} (HTTP {r.status_code})")
        return False
    return True

def extract_gzip(gzip_path, extracted_path):
    print(f"📦 Extracting {gzip_path}")
    try:
        with gzip.open(gzip_path, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"✅ Extracted to {extracted_path}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    return True

def extract_timestamp_and_hour(filename):
    match = re.search(r'pageviews-(\d{4})(\d{2})(\d{2})-(\d{2})', filename)
    if match:
        year, month, day, hour = match.groups()
        timestamp = pd.Timestamp(f"{year}-{month}-{day} {hour}:00:00")
        hour_int = int(hour)
        return timestamp, hour_int
    else:
        raise ValueError(f"❌ Cannot extract timestamp from {filename}")

def safe_remove(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑️ Removed {filepath}")
    else:
        print(f"⚠️ Skipped delete: {filepath} not found")

def generate_unique_id(row):
    """Generate a unique ID based on the row data"""
    # Create a string by combining multiple columns to ensure uniqueness
    unique_string = f"{row['country']}_{row['device']}_{row['title']}_{row['timestamp']}_{row['hour']}"
    # Create a hash of this string to get a fixed-length ID
    return hashlib.md5(unique_string.encode()).hexdigest()

# ---------- PIPELINE ----------
first_chunk = True
row_count = 0  # Global counter for all rows

for hour in HOURS:
    hour_str = f"{hour:02d}"
    fname_gz = f"pageviews-{YEAR}{MONTH:02d}{DAY:02d}-{hour_str}0000.gz"
    url = f"{BASE_URL}/{YEAR}/{YEAR}-{MONTH:02d}/{fname_gz}"
    local_gz = fname_gz
    local_txt = fname_gz.replace('.gz', '')

    # Step 1: Download
    if not download_file(url, local_gz):
        continue

    # Step 2: Extract
    if not extract_gzip(local_gz, local_txt):
        safe_remove(local_gz)
        continue

    # Step 3: Check file size
    if os.path.getsize(local_txt) == 0:
        print(f"⚠️ Warning: extracted file {local_txt} is empty, skipping.")
        safe_remove(local_gz)
        safe_remove(local_txt)
        continue

    # Step 4: Process in chunks
    try:
        timestamp, hour_value = extract_timestamp_and_hour(local_txt)
        chunks = pd.read_csv(local_txt, sep=' ', names=['country_device', 'title', 'views', 'extra'], chunksize=CHUNK_SIZE)

        for chunk in chunks:
            # Split country and device
            chunk['country_device'] = chunk['country_device'].astype(str)
            chunk['country'] = chunk['country_device'].str.split('.').str[0]
            chunk['device'] = chunk['country_device'].str.split('.').str[1]

            # Clean data
            chunk['country'] = chunk['country'].astype(str).str.strip()
            chunk['device'] = chunk['device'].astype(str).str.strip()
            chunk['title'] = chunk['title'].astype(str).str.strip()

            # Filter out bad rows
            chunk = chunk[(chunk['country'] != "nan") & (chunk['title'] != "nan") & (chunk['device'] != "nan")]
            chunk = chunk[(chunk['country'].str.len() > 0) & (chunk['title'].str.len() > 0)]

            # Add timestamp and hour
            chunk['timestamp'] = timestamp
            chunk['hour'] = hour_value
            
            # Add unique IDs
            # Method 1: Generate deterministic IDs based on content
            chunk['id'] = chunk.apply(generate_unique_id, axis=1)
            
            # Method 2: Alternative - use sequential IDs
            # chunk['id'] = range(row_count, row_count + len(chunk))
            # row_count += len(chunk)

            # Keep only needed columns
            chunk = chunk[['id', 'country', 'device', 'title', 'timestamp', 'hour']]

            # Write to CSV
            chunk.to_csv(OUTPUT_CSV, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
            first_chunk = False

    except Exception as e:
        print(f"❌ Error processing {local_txt}: {e}")

    # Step 5: Cleanup
    safe_remove(local_gz)
    safe_remove(local_txt)


