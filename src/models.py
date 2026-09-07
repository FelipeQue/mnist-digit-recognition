"""
Módulo para inicialização e treinamento dos modelos de Machine Learning.
"""

import logging
from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_knn(n_neighbors: int = 5, weights: str = "distance") -> KNeighborsClassifier:
    """Instancia o modelo KNN com os hiperparâmetros especificados.

    Parâmetros
    ----------
    n_neighbors : int, default=5
        Número de vizinhos mais próximos.
    weights : str, default='distance'
        Função de peso na votação ('uniform' ou 'distance').

    Retorna
    -------
    KNeighborsClassifier
        Instância do classificador KNN.
    """
    logger.info(f"Criando KNN com n_neighbors={n_neighbors} e weights='{weights}'")
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )