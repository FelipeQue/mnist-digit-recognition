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

OUTPUTS_DIR = ROOT_DIR / "outputs"

IMAGES_OUTPUT_DIR = OUTPUTS_DIR / "images"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20