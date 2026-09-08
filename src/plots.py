"""
Módulo de plotagem de gráficos e imagens para o projeto Cernere.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional
from src.config import IMAGES_OUTPUT_DIR

logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def plot_digit_samples(
    X: np.ndarray,
    y: np.ndarray,
    filename: str = "digit_first_samples.png",
    save_dir: Optional[Path] = IMAGES_OUTPUT_DIR
) -> None:
    """Plota e salva uma grade 2x5 com a primeira ocorrência de cada dígito (0 a 9).
    Parâmetros:
    X : np.ndarray (n_samples, 784)
        Matriz de pixels das imagens.
    y : np.ndarray (n_samples,)
        Vetor de rótulos correspondentes.
    filename : str, default "digit_first_samples.png"
        Nome do arquivo de saída.
    save_dir : Path, default IMAGES_OUTPUT_DIR
        Diretório para salvar a imagem.
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
    axes = axes.flatten()

    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        image = X[idx].reshape(28, 28)

        axes[digit].imshow(image, cmap="gray_r")
        axes[digit].set_title(f"Dígito: {digit}", fontsize=12, fontweight="regular")
        axes[digit].axis("off")

    logger.info("Gráfico de amostras de dígitos criado com sucesso.")

    plt.suptitle(
        "Primeiras ocorrências dos dígitos de 0 a 9 no MNIST",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    if save_dir:
        save_path = save_dir / filename
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Imagem salva com sucesso em: {save_path}")

    plt.show()
    plt.close()

def plot_confusion_matrices(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    figsize: tuple = (22, 6)
) -> None:
    """Renderiza heatmaps 10x10 da matriz de confusão para múltiplos modelos.

    Parâmetros
    ----------
    predictions_dict : Dict[str, np.ndarray]
        Dicionário mapeando o nome do modelo ao seu vetor de predições.
    y_true : np.ndarray
        Vetor com os rótulos reais de teste.
    figsize : tuple, default=(22, 6)
        Dimensões da figura do Matplotlib.
    """
    from sklearn.metrics import confusion_matrix

    num_models = len(predictions_dict)
    fig, axes = plt.subplots(1, num_models, figsize=figsize)

    if num_models == 1:
        axes = [axes]

    fig.suptitle(
        "Matrizes de Confusão 10x10 no conjunto de teste",
        fontsize=16,
        fontweight="bold"
    )

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=range(10),
            yticklabels=range(10)
        )
        ax.set_title(f"Modelo: {name}", fontsize=14)
        ax.set_xlabel("Dígito previsto (Predicted)", fontsize=12)
        ax.set_ylabel("Dígito real (True)", fontsize=12)

    plt.tight_layout()
    plt.show()