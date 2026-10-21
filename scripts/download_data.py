"""Download the ULB data from the public mirror used in TensorFlow's tutorial."""
from pathlib import Path
import hashlib
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/raw/creditcard.csv"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

if __name__ == "__main__":
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PATH.exists():
        temp = PATH.with_suffix(".download")
        print("Downloading about 144 MB. Please wait.")
        urlretrieve(URL, temp)
        temp.replace(PATH)
    print(PATH)
    print("SHA256:", hashlib.sha256(PATH.read_bytes()).hexdigest())
