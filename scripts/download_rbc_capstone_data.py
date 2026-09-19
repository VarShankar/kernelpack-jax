"""Download and verify the shared public IBAMR RBC trajectory."""

from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile


URL = "https://github.com/VarShankar/kernelpack-matlab/releases/download/data-v1/kernelpack-matlab-rbc-data-v1.zip"
EXPECTED_SHA256 = "1fe40d02155b2ab9cb041211cbf4929d40ce6b382f3bd56fcdcb575ab8466344"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "data" / "ibamr_rbc_3d" / "trajectory"
    if destination.is_dir():
        print(destination)
        return
    with TemporaryDirectory() as temporary:
        archive = Path(temporary) / "rbc.zip"
        urlretrieve(URL, archive)
        digest = sha256(archive.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"RBC data checksum mismatch: {digest}")
        with ZipFile(archive) as zipped:
            zipped.extractall(temporary)
        source = Path(temporary) / "trajectory"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(destination)


if __name__ == "__main__":
    main()
