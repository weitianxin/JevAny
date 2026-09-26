"""Bundled starter data and reproducible public-source mixture builders."""
from importlib.resources import files
from pathlib import Path
import shutil


def init_starter(output_dir: str | Path) -> Path:
    """Copy the bundled, original teaching dataset without network access."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for asset in files(__package__).joinpath("starter").iterdir():
        with asset.open("rb") as source, (destination / asset.name).open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination
