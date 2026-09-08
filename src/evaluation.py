"""Módulo para avaliação de desempenho e cálculo de métricas dos modelos."""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

def generate_predictions(
    model: object,
    X_test: np.ndarray,
    is_keras: bool = False
) -> tuple[np.ndarray, float]:
    """Gera predições para um modelo e mensura o tempo total de inferência.

    Parâmetros
    ----------
    model : object
        Instância do modelo treinado (KNN, XGBoost ou Keras Sequential).
    X_test : np.ndarray
        Matriz de características do conjunto de teste.
    is_keras : bool, default=False
        Flag indicando se o modelo retorna probabilidades que exigem argmax.

    Retorna
    -------
    tuple[np.ndarray, float]
        Vetor com os rótulos previstos e tempo de inferência em segundos.
    """
    start_time = time.perf_counter()

    if is_keras:
        y_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)
    else:
        y_pred = model.predict(X_test)

    infer_time = time.perf_counter() - start_time
    return y_pred, infer_time


def evaluate_models(
    models: dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_times: dict[str, float],
    output_csv: Path | str | None = None
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Calcula métricas de desempenho e consolida os resultados em um DataFrame.

    Parâmetros
    ----------
    models : dict[str, object]
        Dicionário mapeando os nomes dos modelos para suas instâncias.
    X_test : np.ndarray
        Matriz de características do conjunto de teste.
    y_test : np.ndarray
        Vetor de rótulos reais do conjunto de teste.
    train_times : dict[str, float]
        Dicionário contendo os tempos de treinamento registrados.
    output_csv : Path | str | None, default=None
        Caminho para salvar a tabela comparativa em CSV. Se None, não salva.

    Retorna
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        Tabela com métricas consolidadas e dicionário com as predições.
    """
    results_list = []
    predictions = {}

    for name, model in models.items():
        # Verificação flexível para Keras ou detecção via string de nome
        is_keras = "keras" in name.lower() or hasattr(model, "predict_proba")
        
        y_pred, infer_time = generate_predictions(model, X_test, is_keras=is_keras)
        predictions[name] = y_pred

        results_list.append({
            "Modelo": name,
            "Acurácia Global": accuracy_score(y_test, y_pred),
            "Precisão (Weighted)": precision_score(y_test, y_pred, average="weighted"),
            "Recall (Weighted)": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score (Weighted)": f1_score(y_test, y_pred, average="weighted"),
            "Tempo Treino (s)": train_times.get(name, 0.0),
            "Tempo Inferência (s)": infer_time
        })

    df_metrics = pd.DataFrame(results_list)

    if output_csv is not None:
        save_path = Path(output_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(save_path, index=False)
        logger.info("Tabela de métricas salva em: %s", save_path)

    return df_metrics, predictions


def get_top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 3
) -> list[tuple[tuple[int, int], int]]:
    """Identifica os pares de dígitos (verdadeiro, previsto) com maior taxa de erro.

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor com os rótulos reais.
    y_pred : np.ndarray
        Vetor com os rótulos previstos.
    top_n : int, default=3
        Quantidade de pares com maior ocorrência de erro a retornar.

    Retorna
    -------
    list[tuple[tuple[int, int], int]]
        Lista de tuplas contendo ((dígito_real, dígito_previsto), contagem_erros).
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)

    confusions = [
        ((i, j), int(cm[i, j]))
        for i in range(10)
        for j in range(10)
        if cm[i, j] > 0
    ]

    confusions.sort(key=lambda item: item[1], reverse=True)
    return confusions[:top_n]


def format_top_confusions_summary(
    predictions_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    top_n: int = 3
) -> str:
    """Formata os maiores erros de confusão de cada modelo em texto.

    Retorna
    -------
    str
        Texto formatado contendo o resumo dos maiores erros por modelo.
    """
    lines = ["Maiores Taxas de Confusão por Modelo (Verdadeiro vs Previsto)"]
    lines.append("=" * 65)

    for name, y_pred in predictions_dict.items():
        top_errs = get_top_confusions(y_true, y_pred, top_n=top_n)
        err_str = ", ".join(
            [f"Dígito {t} previsto como {p} ({c}x)" for (t, p), c in top_errs]
        )
        lines.append(f"* {name}: {err_str}")

    return "\n".join(lines)


def print_classification_reports(
    predictions_dict: dict[str, np.ndarray],
    y_true: np.ndarray
) -> None:
    """Exibe o relatório detalhado de classificação por classe para cada modelo."""
    for name, y_pred in predictions_dict.items():
        print(f"\n--- Classification Report Detalhado: {name} ---")
        print(classification_report(y_true, y_pred, digits=4))