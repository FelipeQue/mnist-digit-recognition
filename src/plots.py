"""
Módulo de plotagem de gráficos e imagens para o projeto Cernere.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from src.config import IMAGES_OUTPUT_DIR
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def _save_figure(
    fig: plt.Figure,
    save_dir: Path | str | None,
    filename: str,
    dpi: int = 300
) -> None:
    """Função utilitária privada para salvamento de imagens.

    Parâmetros
    ----------
    fig : plt.Figure
        Objeto da figura do Matplotlib a ser salva.
    save_dir : Path | str | None
        Diretório onde a imagem será salva. Se None, a função encerra sem ação.
    filename : str
        Nome do arquivo de saída (ex: 'confusion_matrix_knn.png').
    dpi : int, default=300
        Resolução da imagem salva.
    """
    if save_dir is None:
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    full_path = save_path / filename

    fig.savefig(full_path, bbox_inches="tight", dpi=dpi)
    logger.info("Imagem salva com sucesso em: %s", full_path)

def plot_digit_samples(
    X: np.ndarray,
    y: np.ndarray,
    filename: str = "digit_first_samples.png",
    save_dir: Path | str | None = IMAGES_OUTPUT_DIR
) -> None:
    """Plota e salva uma grade 2x5 com a primeira ocorrência de cada dígito (0 a 9).
    Parâmetros:
    X : np.ndarray (n_samples, 784)
        Matriz de pixels das imagens.
    y : np.ndarray (n_samples,)
        Vetor de rótulos correspondentes.
    filename : str, default "digit_first_samples.png"
        Nome do arquivo de saída.
    save_dir : Path | str | None, default IMAGES_OUTPUT_DIR
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

    plt.tight_layout()
    plt.suptitle(
        "Primeiras ocorrências dos dígitos de 0 a 9 no MNIST",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    
    _save_figure(plt.gcf(), save_dir, filename)

    plt.show()
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path | str | None = IMAGES_OUTPUT_DIR,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = "Blues",
    dpi: int = 300
) -> None:
    """Renderiza e salva a matriz de confusão 10x10 para um único modelo.

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor com os rótulos reais de teste.
    y_pred : np.ndarray
        Vetor com os rótulos previstos pelo modelo.
    model_name : str
        Nome do modelo para exibição no título e formatação do nome do arquivo.
    save_dir : Path | str | None, default=IMAGES_OUTPUT_DIR
        Diretório onde a imagem será salva. Se None, a figura não é salva em disco.
    ax : plt.Axes | None, default=None
        Eixo do Matplotlib para renderização em subplots. Se None, cria uma nova figura.
    figsize : tuple[int, int], default=(6, 5)
        Tamanho da figura (utilizado apenas se ax for None).
    cmap : str, default="Blues"
        Paleta de cores para o heatmap do Seaborn.
    dpi : int, default=300
        Resolução da imagem salva em disco.
    """
    cm = confusion_matrix(y_true, y_pred)

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        ax=ax,
        xticklabels=range(10),
        yticklabels=range(10)
    )

    ax.set_title(f"Matriz de Confusão: {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dígito previsto (Predicted)", fontsize=11)
    ax.set_ylabel("Dígito real (True)", fontsize=11)

    # Padronização e sanitização do nome do arquivo
    file_stub = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"confusion_matrix_{file_stub}.png"

    _save_figure(fig, save_dir, filename, dpi=dpi)

    if show_plot:
        plt.tight_layout()
        plt.show()
        plt.close(fig)