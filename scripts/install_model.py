"""Install fine-tuned GEC model archive onto the local laptop.

Requirements from Phase M3:
- Unzips gec-model.zip into model/gec-model/ on the user's laptop.
- Verifies model files integrity (config, weights, tokenizer).
- Prepares app/engine.py to serve the high-capacity model alongside rule layer.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


DEFAULT_DEST_DIR = Path("model/gec-model")


def find_zip_archive(specified_path: str | None = None) -> Path | None:
    """Search for gec-model.zip in standard download and local locations."""
    if specified_path:
        p = Path(specified_path)
        if p.exists() and p.is_file():
            return p
        print(f"[ERROR] Specified file not found: {specified_path}")
        return None

    candidate_locations = [
        Path("gec-model.zip"),
        Path("model/gec-model.zip"),
        Path.home() / "Downloads" / "gec-model.zip",
        Path.home() / "Desktop" / "gec-model.zip",
    ]

    for cand in candidate_locations:
        if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
            return cand

    return None


def verify_model_files(dest_dir: Path) -> bool:
    """Verify that required model and tokenizer artifacts exist in dest_dir."""
    required_files = ["config.json", "tokenizer_config.json"]
    weight_candidates = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]

    missing_required = [f for f in required_files if not (dest_dir / f).exists()]
    has_weights = any((dest_dir / w).exists() for w in weight_candidates)

    if missing_required:
        print(f"[WARNING] Missing expected files in {dest_dir}: {missing_required}")
        return False

    if not has_weights:
        print(f"[WARNING] No model weights ({weight_candidates}) found in {dest_dir}.")
        return False

    # List all files and total size
    total_bytes = 0
    installed_files = []
    for item in dest_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size
            total_bytes += size
            installed_files.append((item.name, size))

    print("\nVerified Installed Model Files:")
    for fname, size in sorted(installed_files):
        print(f"  * {fname:32s}: {size / (1024 * 1024):>8.2f} MB")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal Model Footprint: {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")
    return True


def install_model():
    parser = argparse.ArgumentParser(description="Install gec-model.zip into model/gec-model/")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to gec-model.zip archive")
    parser.add_argument("--dest-dir", type=str, default=str(DEFAULT_DEST_DIR), help="Destination folder for unpacked model")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)

    print("=" * 80)
    print("GRAMMAR AUTOCORRECTOR: MODEL INSTALLATION UTILITY")
    print("=" * 80)

    zip_path = find_zip_archive(args.zip_path)
    if not zip_path:
        print("\n[ERROR] Could not find 'gec-model.zip'.")
        print("\nHow to install your model:")
        print("  1. Download 'gec-model.zip' from Google Colab or Kaggle.")
        print("  2. Place it in this project folder (or in your Downloads folder).")
        print("  3. Re-run: python scripts/install_model.py")
        print("     Or pass the exact path: python scripts/install_model.py --zip-path /path/to/gec-model.zip")
        print("=" * 80)
        sys.exit(1)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nFound archive: {zip_path.resolve()}")
    print(f"Archive size:  {zip_size_mb:.2f} MB ({zip_size_mb / 1024:.2f} GB)")

    # Prepare destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting archive into '{dest_dir}'...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Archive contains {len(namelist)} items.")

        # Check if archive has an outer folder like "gec-model/..."
        has_outer_dir = any("/" in name for name in namelist) and all(name.startswith("gec-model/") or name == "gec-model" for name in namelist if not name.startswith("__MACOSX"))

        if has_outer_dir:
            print("Detected outer 'gec-model/' folder inside archive. Extracting contents directly...")
            for member in zf.infolist():
                if member.filename.startswith("__MACOSX"):
                    continue
                # Strip prefix
                rel_path = member.filename[len("gec-model/"):]
                if not rel_path:
                    continue
                target_path = dest_dir / rel_path
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
        else:
            zf.extractall(dest_dir)

    print(f"[OK] Extraction complete.")

    # Verify installation
    ok = verify_model_files(dest_dir)
    if ok:
        print("\n" + "=" * 80)
        print("[SUCCESS] Model successfully installed into 'model/gec-model/'!")
        print("=" * 80)
        print("The FastAPI engine (app/engine.py) will now automatically detect and load")
        print("this model on startup, operating in full hybrid mode ('rules + model').")
        print("\nTo start your local server:")
        print("  Windows: run.bat")
        print("  Linux:   ./run.sh")
        print("=" * 80)
    else:
        print("\n[WARNING] Extraction completed but some expected files were missing.")
        sys.exit(1)


if __name__ == "__main__":
    install_model()
