"""
Módulo de preparação e divisão de dados para o projeto Cernere.
"""

from typing import Tuple
import logging
import numpy as np
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