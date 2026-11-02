"""API de consulta do modelo de risco de abandono (CRISP-DM — Implantação).

Executar (na raiz do projeto, depois de `python run_all.py`):
    uvicorn api.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs

O Beready chamaria POST /prever/abandono quando o estudante sai do diagnóstico inicial. Se o risco for alto, o site
dispara uma ação de retenção (ex.: convite para a primeira aula, lembrete no dia seguinte) — a ser validada por teste A/B.
"""
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODELO_PATH = ROOT / "models" / "modelo_abandono.joblib"
INICIO_COLETA = datetime(2018, 8, 1, tzinfo=timezone.utc)

app = FastAPI(title="Beready — API de risco de abandono",
              description="Modelo HistGradientBoosting treinado no EdNet-KT3 (297.915 estudantes). "
                          "Devolve a probabilidade de o estudante não voltar depois do diagnóstico inicial.",
              version="1.0")
_cache = {}


def carregar():
    if "m" not in _cache:
        if not MODELO_PATH.exists():
            raise HTTPException(503, "Modelo não encontrado. Rode antes: python src/ml_ednet_abandono.py")
        _cache["m"] = joblib.load(MODELO_PATH)
    return _cache["m"]


class Diagnostico(BaseModel):
    questoes_respondidas: int = Field(..., ge=0, le=200, description="Questões respondidas no diagnóstico",
                                      examples=[6])
    acertos: int = Field(..., ge=0, le=200, description="Acertos no diagnóstico", examples=[3])
    eventos: int = Field(..., ge=0, le=2000, description="Ações registradas no diagnóstico (enter/respond/submit)",
                         examples=[18])
    inicio: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                             description="Data/hora de início do diagnóstico (ISO 8601)")

    @model_validator(mode="after")
    def coerente(self):
        if self.acertos > self.questoes_respondidas:
            raise ValueError("acertos não pode ser maior que questões_respondidas")
        return self


class Previsao(BaseModel):
    probabilidade_abandono: float
    risco: str
    limiar: float
    acao_sugerida: str
    versao_modelo: str


def atributos(d: Diagnostico) -> pd.DataFrame:
    """Mesma preparação de src/ml_ednet_abandono.py (seção 2)."""
    ts = d.inicio if d.inicio.tzinfo else d.inicio.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    q = d.questoes_respondidas
    meses = (ts.year - INICIO_COLETA.year) * 12 + ts.month - INICIO_COLETA.month
    return pd.DataFrame([{
        "questoes_diag": q, "acertos_diag": d.acertos,
        "acuracia_diag": d.acertos / q if q else float("nan"),
        "eventos_diag": d.eventos, "eventos_por_questao_diag": d.eventos / q if q else float("nan"),
        "pulou_diagnostico": int(q == 0), "hora_inicio": ts.hour, "dia_semana": ts.weekday(),
        "mes_coorte": min(max(meses, 0), 15),  # fora do período de treino: usa a última coorte conhecida (nov/2019)
    }])


@app.get("/saude")
def saude():
    return {"status": "ok", "modelo_disponivel": MODELO_PATH.exists()}


@app.get("/modelo")
def modelo():
    m = carregar()
    return {"descricao": m["descricao"], "versao": m["versao"], "atributos": m["atributos"],
            "limiar": round(m["limiar"], 3), "roc_auc_teste": m["auc_teste"]}


@app.post("/prever/abandono", response_model=Previsao)
def prever(d: Diagnostico):
    m = carregar()
    X = atributos(d)[m["atributos"]]
    p = float(m["modelo"].predict_proba(X)[0, 1])
    alto = p >= m["limiar"]
    acao = ("Risco alto: convidar para a primeira aula logo após o diagnóstico e enviar lembrete em 24 h."
            if alto else "Risco baixo: seguir o fluxo normal.")
    return Previsao(probabilidade_abandono=round(p, 4), risco="alto" if alto else "baixo",
                    limiar=round(m["limiar"], 3), acao_sugerida=acao, versao_modelo=m["versao"])
