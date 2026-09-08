"""Módulo para avaliação e cálculo de métricas dos modelos."""

import logging
import time
from typing import Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def generate_predictions(
    model: Any,
    X_test: np.ndarray,
    is_keras: bool = False
) -> Tuple[np.ndarray, float]:
    """Gera predições para um modelo e mensura o tempo de inferência.

    Parâmetros
    ----------
    model : Any
        Instância do modelo treinado (KNN, XGBoost ou Keras Sequential).
    X_test : np.ndarray
        Dados de entrada para inferência.
    is_keras : bool, default=False
        Flag indicando se o modelo necessita de extração via argmax.

    Retorna
    -------
    Tuple[np.ndarray, float]
        Vetor com os rótulos preditos e o tempo total de inferência em segundos.
    """
    start_time = time.time()

    if is_keras:
        y_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)
    else:
        y_pred = model.predict(X_test)

    infer_time = time.time() - start_time
    return y_pred, infer_time