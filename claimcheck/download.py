"""Download the SciFact dataset into data/scifact/.

Run once:  python -m claimcheck.download
"""
import io
import tarfile
from pathlib import Path

import requests

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        # filter="data" refuses absolute paths, "..", links and device files
        archive.extractall(DATA_DIR, filter="data")
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size:>10,}  {path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
