import torch
import os
from pathlib import Path

# Imposta la working directory alla root del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)

def check_environment():
    print("🚀 AlienNet - Environment Check")
    print("=" * 40)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ PyTorch device available: {device}")

    # Check folders
    folders = [
        "models", "adversarial", "training", "utils", "notebooks", "data", "experiments"
    ]
    for folder in folders:
        path = Path(folder)
        if path.exists() and path.is_dir():
            print(f"📁 Found folder: {folder}")
        else:
            print(f"⚠️  Missing folder: {folder}")

    # Test tensor
    x = torch.rand(2, 3)
    print(f"\n🧪 Test tensor: shape {x.shape} – values:\n{x}")

    print("\n✅ Environment appears ready.\n")

if __name__ == "__main__":
    check_environment()
