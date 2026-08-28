"""Download the BigMart Sales dataset from Kaggle into data/raw/.

Requires a Kaggle API token to be configured on the machine — either
~/.kaggle/access_token (new token format) or ~/.kaggle/kaggle.json
(legacy username+key format). See README.md for setup instructions.
"""

import shutil
from pathlib import Path

import kagglehub

from src.config import ROOT_DIR, load_params


def download_dataset() -> Path:
    params = load_params()
    dataset = params["kaggle"]["dataset"]
    raw_dir = ROOT_DIR / params["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download(dataset))

    copied = []
    for csv_file in cache_path.glob("*.csv"):
        dest = raw_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        copied.append(dest)

    if not copied:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset at {cache_path}")

    print(f"Copied {len(copied)} file(s) to {raw_dir}:")
    for f in copied:
        print(f"  - {f.name}")

    return raw_dir


if __name__ == "__main__":
    download_dataset()
