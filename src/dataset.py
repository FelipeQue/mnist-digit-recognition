"""
Módulo de carregamento e/ou manipulação do dataset MNIST.
"""

import logging
import numpy as np
from sklearn.datasets import fetch_openml
from src.config import MNIST_RAW_PATH

logger = logging.getLogger(__name__)

def load_mnist_data():
    """
    Carrega o dataset MNIST. Se já existir em cache local, lê do disco.
    Caso contrário, baixa do OpenML e salva em data/raw/mnist_784.npz.
    """
    if MNIST_RAW_PATH.exists():
        logger.info("Carregando MNIST do cache local: %s", MNIST_RAW_PATH)
        with np.load(MNIST_RAW_PATH) as data:
            X = data["X"]
            y = data["y"]
        logger.info("Dataset carregado com sucesso do cache local.")
        return X, y
    
    logger.info("Baixando dataset MNIST do OpenML (primeira execução)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target.astype(np.uint8)
    
    np.savez_compressed(MNIST_RAW_PATH, X=X, y=y)
    logger.info("Dataset salvo com sucesso em: %s", MNIST_RAW_PATH)
    
    return X, y