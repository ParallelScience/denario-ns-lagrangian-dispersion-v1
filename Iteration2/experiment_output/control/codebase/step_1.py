# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_file(idx):
    data_dir = "data/"
    filename = "Turb.hydro_w." + str(idx) + ".vtk"
    filepath = os.path.join(data_dir, filename)
    url = "https://huggingface.co/datasets/pedrota2000/NS_simulation/resolve/main/" + filename
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response, open(filepath, 'wb') as out_file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return True
    except Exception as e:
        print("Error downloading " + url + ": " + str(e))
    return False

def download_snapshots():
    indices = list(range(18903, 19899, 5))
    downloaded_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_file, idx) for idx in indices]
        for future in as_completed(futures):
            if future.result():
                downloaded_count += 1
    print("Successfully downloaded/verified files: " + str(downloaded_count) + " out of 200")

if __name__ == '__main__':
    download_snapshots()