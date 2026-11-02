"""Funções compartilhadas da etapa de Modelagem com aprendizado de máquina (2º bimestre).

Reaproveita a paleta e o estilo de `utils.py` (mesma identidade visual dos gráficos do 1º bimestre) e acrescenta:
- cálculo padronizado das métricas de classificação;
- utilitários de figura com Seaborn;
- serialização dos resultados em JSON.
"""
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from utils import ARES, EDNET, INK, INK2, MUTED, NEG, NEUTRAL, POS, SEQ, save  # noqa: F401  (estilo compartilhado)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # sem avisos de fonte fora do Windows

MODEL_COLORS = {"Baseline (classe majoritária)": NEUTRAL, "Regressão Logística": "#2a78d6",
                "Árvore de Decisão": "#7a5cc4", "Random Forest": "#2e9e6a", "Gradient Boosting (HGB)": "#eb6834"}


def metricas(y, proba, limiar=0.5, positivo="classe 1"):
    """Métricas de classificação binária a partir das probabilidades da classe positiva."""
    pred = (proba >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "limiar": round(float(limiar), 4),
        "acuracia": accuracy_score(y, pred),
        "acuracia_balanceada": balanced_accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "especificidade": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "matriz_confusao": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
        "classe_positiva": positivo,
        "n": int(len(y)),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def limiar_youden(y, proba):
    """Limiar que maximiza a acurácia balanceada (índice de Youden = sensibilidade + especificidade − 1)."""
    grid = np.linspace(0.05, 0.95, 181)
    ba = [balanced_accuracy_score(y, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(ba))])


class Cronometro:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = round(time.perf_counter() - self.t, 1)


def resumo_cv(cv_res):
    """Média e desvio-padrão de cada métrica de um `cross_validate`."""
    out = {}
    for k, v in cv_res.items():
        if k.startswith("test_"):
            out[k[5:]] = {"media": round(float(np.mean(v)), 4), "dp": round(float(np.std(v)), 4)}
    out["tempo_ajuste_s"] = round(float(np.sum(cv_res["fit_time"])), 1)
    return out


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.ndarray, pd.Series)):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%".replace(".", ",")
