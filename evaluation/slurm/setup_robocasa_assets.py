#!/usr/bin/env python3
"""Download simulator assets, with markers to avoid repeated multi-GB downloads."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from zipfile import ZipFile

from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY, download_url

ROOT = Path(__file__).resolve().parents[2]
markers = ROOT / ".cache/robocasa-assets"
markers.mkdir(parents=True, exist_ok=True)
# Optional local source must contain the same RoboCasa v0.2 asset layout.
source_root = os.environ.get("XWAM_ROBOCASA_ASSETS")


def copy_file(source, destination):
    source_stat = os.stat(source)
    try:
        destination_stat = os.stat(destination)
        if (source_stat.st_size, source_stat.st_mtime_ns) == (destination_stat.st_size, destination_stat.st_mtime_ns):
            return str(destination)
    except FileNotFoundError:
        pass
    return shutil.copy2(source, destination)


def copy_assets(source, folder):
    folder.mkdir(parents=True, exist_ok=True)

    def copy_entry(entry):
        destination = folder / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True, copy_function=copy_file)
        else:
            copy_file(entry, destination)

    # Native assets contain many small files; independent directories can copy
    # concurrently on shared storage. Completed files are reused on reruns.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_entry, source.iterdir()))


for name, config in DOWNLOAD_ASSET_REGISTRY.items():
    if name == "aigen_objs":
        continue  # Upstream also skips this unused collection.
    folder = Path(config["folder"])
    marker = markers / f"{name}.complete"
    if marker.is_file() and folder.is_dir():
        print(f"Assets already installed: {name}")
        continue
    if source_root:
        relative = folder.relative_to(ROOT / "third_party/robocasa/robocasa/models/assets")
        source = Path(source_root) / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        copy_assets(source, folder)
    else:
        # The upstream extraction helper suppresses failed downloads. Propagate
        # failures here so existing partial asset folders cannot appear ready.
        archive = markers / f"{name}.zip"
        for attempt in range(3):
            try:
                download_url(config["url"], str(markers), fname=archive.name, check_overwrite=False)
                break
            except Exception:
                if attempt == 2:
                    raise
        folder.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zipped:
            zipped.extractall(folder.parent)
        archive.unlink()
    if not folder.is_dir() or not any(folder.iterdir()):
        raise RuntimeError(f"Asset download failed: {name}")
    marker.write_text(config["url"] + "\n")
    print(f"Installed assets: {name}", flush=True)
