"""Testes da API (python -m pytest tests/ ou python tests/test_api.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

c = TestClient(app)


def test_saude():
    assert c.get("/saude").json()["status"] == "ok"


def test_parou_na_sexta_questao_tem_risco_alto():
    r = c.post("/prever/abandono", json={"questoes_respondidas": 6, "acertos": 3, "eventos": 18,
                                         "inicio": "2019-03-10T14:00:00Z"}).json()
    assert r["risco"] == "alto" and r["probabilidade_abandono"] > 0.9


def test_completou_diagnostico_tem_risco_menor():
    r = c.post("/prever/abandono", json={"questoes_respondidas": 11, "acertos": 6, "eventos": 33,
                                         "inicio": "2019-03-10T14:00:00Z"}).json()
    assert r["risco"] == "baixo" and r["probabilidade_abandono"] < 0.3


def test_validacao_entrada():
    assert c.post("/prever/abandono", json={"questoes_respondidas": 5, "acertos": 9, "eventos": 15}).status_code == 422


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
    print(c.post("/prever/abandono", json={"questoes_respondidas": 6, "acertos": 3, "eventos": 18,
                                           "inicio": "2019-03-10T14:00:00Z"}).json())
