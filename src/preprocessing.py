"""
Módulo de preparação e divisão de dados e de processamento de imagens para o projeto Cernere. 
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple
from pathlib import Path
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE

logger = logging.getLogger(__name__)


def split_data(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide os dados em conjuntos de Treino, Validação e Teste com estratificação.

    Proporção padrão definida em config.py (70% Treino, 10% Validação, 20% Teste).

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de atributos (pixels).
    y : np.ndarray
        Vetor de rótulos.

    Retorna
    -------
    Tuple[np.ndarray, ...]
        (X_train, X_val, X_test, y_train, y_val, y_test)

    """

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    relative_val_size = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    logger.info(f"Split realizado com sucesso:")
    logger.info(f" - Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X):.1%})")
    logger.info(f" - Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X):.1%})")
    logger.info(f" - Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X):.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_pixels(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redimensiona os valores de pixels do intervalo [0, 255] para a escala [0.0, 1.0].

    Parâmetros
    ----------
    X_train : np.ndarray
        Dados de treino.
    X_val : np.ndarray
        Dados de validação.
    X_test : np.ndarray
        Dados de teste.

    Retorna
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Conjuntos (X_train_scaled, X_val_scaled, X_test_scaled) no tipo float32.
    """
    # Garante tipo float32
    X_train_scaled = X_train.astype(np.float32) / 255.0
    X_val_scaled = X_val.astype(np.float32) / 255.0
    X_test_scaled = X_test.astype(np.float32) / 255.0

    logger.info("Normalização realizada com sucesso: pixels ajustados para a escala [0.0, 1.0].")
    return X_train_scaled, X_val_scaled, X_test_scaled

def process_custom_digit_to_df(
    image_path: str | Path,
    invert_colors: bool = True
) -> tuple[np.ndarray, pd.DataFrame]:
    """Processa uma imagem customizada para o formato 28x28.

    Aplica conversão para escala de cinza, inversão de cores, recorte
    por bounding box com padding proporcional, alinhamento pelo centro de
    massa e normalização para o intervalo [0.0, 1.0].

    Parâmetros
    ----------
    image_path : str | Path
        Caminho relativo ou absoluto para o arquivo de imagem em data/custom/.
    invert_colors : bool, padrão=True
        Se True, inverte cores (papel branco/caneta escura -> fundo preto/traço branco).

    Retorna
    -------
    tuple[np.ndarray, pd.DataFrame]
        - Matriz NumPy 28x28 normalizada em [0.0, 1.0] pronta para inferência.
        - DataFrame do Pandas (28 linhas x 28 colunas) com as intensidades dos pixels.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")

    # Carrega a imagem e converte para escala de cinza ('L')
    img = Image.open(image_path).convert("L")

    # Inverte cores caso o fundo seja claro
    if invert_colors:
        img = ImageOps.invert(img)

    # Recorte por Bounding Box com Padding proporcional
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Manter proporção original e encaixar em um quadrado 20x20
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Inserir no centro de um canvas 28x28 preto
    canvas = Image.new("L", (28, 28), 0)
    upper_left = ((28 - img.width) // 2, (28 - img.height) // 2)
    canvas.paste(img, upper_left)

    # Normalizar para [0.0, 1.0]
    img_array = np.array(canvas, dtype=np.float32) / 255.0

    # Centralização pelo Centro de Massa (Padrão MNIST)
    cy, cx = center_of_mass(img_array)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = np.round(14.0 - cy)
        shift_x = np.round(14.0 - cx)
        img_array = shift(img_array, shift=(shift_y, shift_x), mode="constant", cval=0.0)

    # Garantir limites [0.0, 1.0] após a interpolação do shift
    img_array = np.clip(img_array, 0.0, 1.0)

    # Estruturar a tabela Pandas (28x28)
    df_pixels = pd.DataFrame(
        img_array,
        index=[f"row_{i}" for i in range(28)],
        columns=[f"col_{j}" for j in range(28)]
    )

    logger.info("Imagem %s processada com sucesso.", image_path.name)
    return img_array, df_pixels