"""
Módulo para inicialização e treinamento dos modelos de Machine Learning.
"""

import logging
from typing import Literal
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

KNNWeightsType = Literal["uniform", "distance"]
XGBEvalMetricType = Literal["mlogloss", "logloss", "error"]
ActivationType = Literal["relu", "tanh", "sigmoid"]

def build_knn(
        n_neighbors: int = 5,
        weights: KNNWeightsType = "distance"
        ) -> KNeighborsClassifier:
    """Instancia o modelo KNN com os hiperparâmetros especificados.

    Parâmetros
    ----------
    n_neighbors : int, default=5
        Número de vizinhos mais próximos.
    weights : KNNWeightsType, default='distance'
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

def build_xgboost(
    n_estimators: int = 100, 
    max_depth: int = 6, 
    learning_rate: float = 0.1, 
    eval_metric: XGBEvalMetricType = "mlogloss",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1
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
    eval_metric : XGBEvalMetricType, default='mlogloss'
        Métrica de avaliação.
    random_state : int, default=RANDOM_STATE
        Semente para reprodutibilidade.

    Retorna
    -------
    XGBClassifier
        Instância configurada do XGBoost.
    """
    logger.info(
        f"Instanciando XGBClassifier com n_estimators={n_estimators}, "
        f"max_depth={max_depth}, learning_rate={learning_rate}, eval_metric={eval_metric}"
    )
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric=eval_metric,
        random_state=random_state,
        tree_method="hist",
        n_jobs=n_jobs,
    )


def create_multilayer_perceptron(
    input_dim: int = 784,
    num_classes: int = 10,
    hidden_units: int = 128,
    activation: ActivationType = "relu",
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2
) -> Sequential:
    """Cria e compila um Perceptron Multicamadas (MLP) desacoplado de otimizadores externos.

    Parâmetros
    ----------
    input_dim : int, default=784
        Número de dimensões de entrada (pixels).
    num_classes : int, default=10
        Número de classes de saída (dígitos de 0 a 9).
    hidden_units : int, default=128
        Número de neurônios na primeira camada oculta.
    activation : ActivationType, default='relu'
        Função de ativação para as camadas ocultas.
    learning_rate : float, default=0.001
        Taxa de aprendizado do otimizador Adam.
    dropout_rate : float, default=0.2
        Taxa de descarte da camada de Dropout para regularização.

    Retorna
    -------
    Sequential
        Modelo Keras compilado e pronto para treinamento.
    """
    logger.info(
        f"Instanciando MLP Classifier [units={hidden_units}, "
        f"activation={activation}, lr={learning_rate}]"
    )

    model = Sequential([
        Dense(hidden_units, activation=activation, input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(hidden_units // 2, activation=activation),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_hyper_mlp(hp) -> Sequential:
    """Função de fábrica adaptada para injeção de parâmetros pelo KerasTuner.

    Parâmetros
    ----------
    hp : keras_tuner.HyperParameters
        Objeto de injeção e amostragem de hiperparâmetros.

    Retorna
    -------
    Sequential
        Modelo instanciado com os hiperparâmetros amostrados.
    """
    hidden_units = hp.Choice("units", values=[64, 128, 256])
    activation = hp.Choice("activation", values=["relu", "tanh"])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    return create_multilayer_perceptron(
        hidden_units=hidden_units,
        activation=activation,
        learning_rate=learning_rate
    )

def create_best_mlp() -> Sequential:
    """Cria e compila a MLP com os melhores hiperparâmetros obtidos na Fase 3."""
    return create_multilayer_perceptron(
        hidden_units=128,
        activation="relu",
        learning_rate=0.001
    )