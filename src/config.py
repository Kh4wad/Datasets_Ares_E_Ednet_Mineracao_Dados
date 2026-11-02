"""Caminhos e parâmetros globais do projeto (CRISP-DM — Beready)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------- ARES (OSF 8y3zp) ----------------
ARES_DIR = ROOT / "data" / "raw" / "ares" / "data"
ARES_RAW = ARES_DIR / "raw"

# ---------------- EdNet (riiid) -------------------
# O KT3 tem 297.915 CSVs (um por estudante): ~763 MB zipado ou ~3,5 GB descompactado. Aceita os dois formatos:
#   KT3:      EdNet-KT3.zip  |  pasta EdNet-KT3/ (com os u*.csv dentro, direto ou em KT3/)  |  pasta KT3/
#   Contents: contents/questions.csv  |  EdNet-Contents/contents/questions.csv  |  EdNet-Contents/questions.csv
# Pastas-base procuradas, nesta ordem:
#   1. variável de ambiente EDNET_DIR
#   2. <projeto>/data/raw/ednet             (padrão; ignorado pelo git)
#   3. <projeto>/Datasets  e  <pasta acima do projeto>/Datasets   (layout "Proj_mineracoa/Datasets")
#   4. <pasta acima do projeto>/Datasets-Beready-raw/ednet  e  <home>/Datasets-Beready-raw/ednet
_BASES = [Path(os.environ["EDNET_DIR"])] if os.environ.get("EDNET_DIR") else []
_BASES += [ROOT / "data" / "raw" / "ednet", ROOT / "Datasets", ROOT.parent / "Datasets",
           ROOT.parent / "Datasets-Beready-raw" / "ednet", Path.home() / "Datasets-Beready-raw" / "ednet"]


def _pasta_kt3(p):
    """Devolve a pasta que contém os u*.csv (p, p/KT3), ou None."""
    for d in (p, p / "KT3"):
        if d.is_dir() and next(d.glob("u*.csv"), None) is not None:
            return d
    return None


def _localizar_ednet():
    for b in _BASES:
        kt3 = None
        if (b / "EdNet-KT3.zip").is_file():
            kt3 = b / "EdNet-KT3.zip"
        else:
            kt3 = next(filter(None, (_pasta_kt3(b / n) for n in ("EdNet-KT3", "KT3"))), None)
        contents = next((d for d in (b / "contents", b / "EdNet-Contents" / "contents", b / "EdNet-Contents")
                         if (d / "questions.csv").is_file()), None)
        if kt3 or contents:
            return b, kt3, contents
    return _BASES[0], None, None


EDNET_DIR, EDNET_KT3, _CONTENTS = _localizar_ednet()
EDNET_KT3_ZIP = EDNET_KT3                     # nome antigo mantido por compatibilidade (pode ser zip OU pasta)
EDNET_CONTENTS = _CONTENTS or EDNET_DIR / "contents"
EDNET_QUESTIONS = EDNET_CONTENTS / "questions.csv"
EDNET_LECTURES = EDNET_CONTENTS / "lectures.csv"
EDNET_BRUTO_DISPONIVEL = EDNET_KT3 is not None and EDNET_QUESTIONS.is_file()

# ---------------- Saídas --------------------------
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for p in (PROCESSED, FIG / "ares", FIG / "ednet", FIG / "comparacao", TAB):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Parâmetros metodológicos --------
SESSION_GAP_MIN = 30          # inatividade > 30 min inicia nova sessão
MAX_CONTENT_MIN = 60          # leitura de explicação/aula > 60 min é descartada (aba esquecida aberta)
EDNET_MIN_ANSWERS = 10        # mínimo de questões respondidas para medir desempenho
EDNET_EVOL_WINDOW = 20        # evolução = acerto nas últimas 20 − primeiras 20 questões
RANDOM_STATE = 42
