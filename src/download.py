"""Step 0 - Download the SciFact dataset.

The HuggingFace page (https://huggingface.co/datasets/allenai/scifact) is backed by a
loading script, scifact.py, which simply downloads the official release tarball from
AllenAI. Recent versions of the `datasets` library refuse to run such scripts, so we
fetch the same tarball directly.

Run:  python -m src.download
"""
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "scifact"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "data.tar.gz"
    if not archive.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, archive)

    OUT_DIR.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            # Keep only the top-level jsonl files; skip macOS "._" metadata files
            # and the cross-validation folds (not needed for this project).
            if member.isfile() and name.endswith(".jsonl") and not name.startswith("._") \
                    and "cross_validation" not in member.name:
                with tar.extractfile(member) as src, open(OUT_DIR / name, "wb") as dst:
                    dst.write(src.read())
                print(f"  extracted {name}")
    print(f"Done. Files are in {OUT_DIR}")


if __name__ == "__main__":
    main()
