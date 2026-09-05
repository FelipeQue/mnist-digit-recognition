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

    Parameters
    ----------
    X : np.ndarray
        Matriz de atributos (pixels).
    y : np.ndarray
        Vetor de rótulos.

    Returns
    -------
    Tuple[np.ndarray, ...]
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 1. Primeira divisão: Separa os 20% de Teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 2. Segunda divisão: Separa o restante (80%) em Treino (70% do total) e Validação (10% do total)
    # Como X_temp representa 80% do total, a proporção de validação sobre X_temp é 0.10 / 0.80 = 0.125
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