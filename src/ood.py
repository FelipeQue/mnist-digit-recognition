"""Módulo para experimentos de Class Masking e Generalização Extrema (Fase 5)."""

import logging
import time
import numpy as np
from collections.abc import Callable
from keras import Sequential

logger = logging.getLogger(__name__)

def filter_masked_dataset(
    X: np.ndarray,
    y: np.ndarray,
    masked_classes: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove completamente as classes especificadas do conjunto de dados.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de características (pixels escalados).
    y : np.ndarray
        Vetor de rótulos.
    masked_classes : list[int]
        Lista de dígitos a serem ocultados do treino.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        Subconjunto de X e y sem as classes ocultadas.
    """
    mask = ~np.isin(y, masked_classes)
    X_filtered, y_filtered = X[mask], y[mask]

    logger.info(
        "Classes %s removidas. Amostras restantes: %d de %d",
        masked_classes,
        len(y_filtered),
        len(y)
    )
    return X_filtered, y_filtered


def train_masked_mlp(
    build_model_fn: Callable[[], Sequential],
    X_train: np.ndarray,
    y_train: np.ndarray,
    masked_classes: list[int],
    epochs: int = 20,
    batch_size: int = 64
) -> tuple[Sequential, float]:
    """Treina uma nova instância da MLP sem visualizar as classes ocultadas.

    Parâmetros
    ----------
    build_model_fn : Callable[[], Sequential]
        Função fábrica que instancia e compila uma nova MLP limpa.
    X_train : np.ndarray
        Matriz de características de treino.
    y_train : np.ndarray
        Vetor de rótulos de treino.
    masked_classes : list[int]
        Dígitos que serão removidos do ajuste de pesos.
    epochs : int, default=20
        Número de épocas de treinamento.
    batch_size : int, default=64
        Tamanho do lote.

    Retorna
    -------
    tuple[Sequential, float]
        Nova instância da MLP treinada e tempo decorrido em segundos.
    """
    X_train_masked, y_train_masked = filter_masked_dataset(
        X_train, y_train, masked_classes
    )

    masked_mlp = build_model_fn()

    start_time = time.perf_counter()
    masked_mlp.fit(
        X_train_masked,
        y_train_masked,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    train_time = time.perf_counter() - start_time
    logger.info("Treinamento restrito concluído em %.2fs", train_time)

    return masked_mlp, train_time