"""Packager les modeles finaux en zip (pour GitHub Release)."""

from __future__ import annotations

from pathlib import Path
import zipfile

BASE_DIR = Path("outputs/final_models")
OUTPUT_ZIP = Path("outputs/final_models_bundle.zip")


def main() -> int:
    if not BASE_DIR.exists():
        print("Dossier outputs/final_models introuvable.")
        return 1

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in BASE_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(BASE_DIR))

    print(f"Zip cree: {OUTPUT_ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
