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

"""
Módulo para inicialização dos modelos de Machine Learning.
"""

import logging
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_xgboost(
    n_estimators: int = 100, 
    max_depth: int = 6, 
    learning_rate: float = 0.1, 
    random_state: int = 42
) -> XGBClassifier:
    """Instancia o modelo XGBoost Classifier.

    Parâmetros
    ----------
    n_estimators : int, default=100
        Número de árvores na sequência de boosting.
    max_depth : int, default=6
        Profundidade máxima de cada árvore.
    learning_rate : float, default=0.1
        Taxa de aprendizado.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    XGBClassifier
        Instância configurada do XGBoost.
    """
    logger.info(
        f"Instanciando XGBClassifier com n_estimators={n_estimators}, "
        f"max_depth={max_depth}, learning_rate={learning_rate}"
    )
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss"
    )