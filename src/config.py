"""
Caminhos e parâmetros centrais do projeto, usados pelo notebook e pelos
módulos de src/.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

MNIST_RAW_PATH = RAW_DATA_DIR / "mnist_784.npz"