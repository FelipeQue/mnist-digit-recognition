
import logging
import numpy as np
from sklearn.datasets import fetch_openml
from src.config import MNIST_RAW_PATH

def load_mnist_data():
    """
    Carrega o dataset MNIST. Se já existir em cache local, lê do disco.
    Caso contrário, baixa do OpenML e salva em data/raw/mnist_784.npz.
    """
    if MNIST_RAW_PATH.exists():
        logging.info(f"Carregando MNIST do cache local: {MNIST_RAW_PATH}")
        data = np.load(MNIST_RAW_PATH)
        return data['X'], data['y']
    
    logging.info("Baixando dataset MNIST do OpenML (primeira execução)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target.astype(np.uint8)
    
    np.savez_compressed(MNIST_RAW_PATH, X=X, y=y)
    logging.info(f"Dataset salvo com sucesso em: {MNIST_RAW_PATH}")
    
    return X, y