"""EdNet-KT3 — Etapas 2 e 3 do CRISP-DM (entendimento + preparação).

Lê os 297.915 CSVs (um por estudante) do zip ou da pasta descompactada, valida e limpa cada arquivo
e gera UMA linha de atributos por estudante em data/processed/ednet_users.csv.gz.
Também registra a contagem global de linhas, duplicadas, nulos e inválidos em
outputs/tables/ednet_qualidade.json.

Uso:  python src/ednet_01_extract.py [--limit N]
"""
import io
import json
import re
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import config as C

VALID_ACTIONS = {"enter", "respond", "submit", "quit"}
VALID_PLATFORMS = {"mobile", "web"}
VALID_ANSWERS = set("abcd")
ITEM_RE = re.compile(r"^[qbel]\d+$")
# Janela de coleta do KT3 (README: desde 27/08/2018; arquivo publicado em dez/2019) com folga
TS_MIN = pd.Timestamp("2017-01-01").value // 10**6
TS_MAX = pd.Timestamp("2020-01-01").value // 10**6
GAP_MS = C.SESSION_GAP_MIN * 60_000
MAX_CONTENT_MS = C.MAX_CONTENT_MIN * 60_000

_zip = None
_answer_key = None


# ---- Fonte dos CSVs: EdNet-KT3.zip ou pasta descompactada (ver src/config.py)
def listar_arquivos():
    """Nomes dos CSVs de estudantes (no zip: 'KT3/u1.csv'; na pasta: 'u1.csv')."""
    if C.EDNET_KT3 is None:
        raise FileNotFoundError(f"EdNet-KT3 não encontrado (zip ou pasta). Pastas procuradas: {C._BASES}")
    if C.EDNET_KT3.is_file():
        with zipfile.ZipFile(C.EDNET_KT3) as z:
            return [n for n in z.namelist() if n.startswith("KT3/") and n.endswith(".csv")]
    return sorted(p.name for p in C.EDNET_KT3.glob("u*.csv"))


def ler_bytes(name):
    if C.EDNET_KT3.is_file():
        global _zip
        if _zip is None:
            _zip = zipfile.ZipFile(C.EDNET_KT3)
        return _zip.read(name)
    return (C.EDNET_KT3 / name).read_bytes()


def ler_estudante(uid):
    """CSV bruto de um estudante (ex.: 'u1'), como texto — para inspeção nos notebooks."""
    name = f"KT3/{uid}.csv" if C.EDNET_KT3.is_file() else f"{uid}.csv"
    return pd.read_csv(io.BytesIO(ler_bytes(name)), dtype=str, keep_default_na=False, na_values=[""])


def _init():
    global _answer_key
    q = pd.read_csv(C.EDNET_QUESTIONS, usecols=["question_id", "correct_answer"])
    _answer_key = dict(zip(q.question_id, q.correct_answer))


def process_user(name):
    q = Counter()  # contadores de qualidade deste arquivo
    df = pd.read_csv(io.BytesIO(ler_bytes(name)), dtype=str, keep_default_na=False, na_values=[""])
    q["linhas_brutas"] += len(df)
    for col in df.columns:
        q[f"nulos_{col}"] += int(df[col].isna().sum())

    # 1) duplicadas exatas
    dup = df.duplicated()
    q["duplicadas"] += int(dup.sum())
    df = df[~dup]

    # 2) validação de domínio
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    bad_ts = ts.isna() | (ts < TS_MIN) | (ts > TS_MAX)
    bad_action = ~df["action_type"].isin(VALID_ACTIONS)
    bad_item = ~df["item_id"].fillna("").str.match(ITEM_RE)
    bad_platform = ~df["platform"].isin(VALID_PLATFORMS)
    is_resp = df["action_type"].eq("respond")
    bad_answer = is_resp & ~df["user_answer"].isin(VALID_ANSWERS)
    bad_resp_item = is_resp & ~df["item_id"].fillna("").str.startswith("q")
    q["invalido_timestamp"] += int(bad_ts.sum())
    q["invalido_action_type"] += int(bad_action.sum())
    q["invalido_item_id"] += int(bad_item.sum())
    q["invalido_platform"] += int(bad_platform.sum())
    q["invalido_resposta"] += int(bad_answer.sum())
    q["invalido_respond_sem_questao"] += int(bad_resp_item.sum())
    bad = bad_ts | bad_action | bad_item | bad_answer | bad_resp_item  # plataforma inválida não invalida o evento
    q["linhas_removidas_invalidas"] += int(bad.sum())
    df = df[~bad].copy()
    df["timestamp"] = ts[~bad].astype("int64")

    # 3) ordenação temporal (registra se o arquivo vinha fora de ordem)
    q["arquivos_fora_de_ordem"] += int(not df["timestamp"].is_monotonic_increasing)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    q["linhas_limpas"] += len(df)

    uid = name.split("/")[-1].removesuffix(".csv")
    if df.empty:
        q["usuarios_vazios"] += 1
        return None, q

    t = df["timestamp"].to_numpy()
    act = df["action_type"].to_numpy()
    item = df["item_id"].to_numpy()
    src = df["source"].fillna("desconhecido").to_numpy()
    prefix = np.array([s[0] for s in item])

    # ---- sessões e tempo de estudo
    gaps = np.diff(t)
    new_sess = np.concatenate([[True], gaps > GAP_MS])
    n_sessions = int(new_sess.sum())
    study_ms = int(gaps[gaps <= GAP_MS].sum())
    days = pd.to_datetime(t, unit="ms").normalize()
    active_days = int(days.nunique())
    span_days = float((t[-1] - t[0]) / 86_400_000)

    row = {
        "user_id": uid,
        "n_eventos": len(df),
        "n_sessoes": n_sessions,
        "dias_ativos": active_days,
        "periodo_dias": round(span_days, 3),
        "tempo_estudo_min": round(study_ms / 60_000, 3),
        "pct_mobile": round(float((df["platform"] == "mobile").mean()), 4),
        "primeiro_evento": int(t[0]),
        "ultimo_evento": int(t[-1]),
    }
    for a in VALID_ACTIONS:
        row[f"acao_{a}"] = int((act == a).sum())

    # ---- tipo de conteúdo acessado (enter): b=bloco de questões, e=explicação, l=aula
    enter = act == "enter"
    for p, nome in (("b", "questoes"), ("e", "explicacao"), ("l", "aula")):
        row[f"acessos_{nome}"] = int((enter & (prefix == p)).sum())

    # tempo em explicações/aulas: pares enter -> quit consecutivos do mesmo item
    for p, nome in (("e", "explicacao"), ("l", "aula")):
        m = prefix == p
        sub_t, sub_a, sub_i = t[m], act[m], item[m]
        dur = 0
        for k in range(len(sub_t) - 1):
            if sub_a[k] == "enter" and sub_a[k + 1] == "quit" and sub_i[k] == sub_i[k + 1]:
                d = sub_t[k + 1] - sub_t[k]
                if 0 <= d <= MAX_CONTENT_MS:
                    dur += d
        row[f"tempo_{nome}_min"] = round(dur / 60_000, 3)

    # ---- eventos por origem (source = contexto da atividade)
    for s, n in Counter(src).items():
        row[f"src_{s}"] = n

    # ---- respostas finais: última resposta de cada questão antes do submit
    seg = np.concatenate([[0], np.cumsum(act[:-1] == "submit")])
    df["seg"] = seg
    submitted = set(seg[act == "submit"])
    resp = df[(df["action_type"] == "respond") & df["seg"].isin(submitted)]
    q["respostas_intermediarias_descartadas"] += int(len(resp) - resp.drop_duplicates(["seg", "item_id"]).shape[0])
    final = resp.drop_duplicates(["seg", "item_id"], keep="last")
    key = final["item_id"].map(_answer_key)
    q["questoes_sem_gabarito"] += int(key.isna().sum())
    final = final[key.notna()]
    correct = (final["user_answer"].to_numpy() == key[key.notna()].to_numpy())
    n_ans = len(final)
    row["n_respondidas"] = n_ans
    row["n_acertos"] = int(correct.sum())
    row["acuracia"] = round(float(correct.mean()), 4) if n_ans else np.nan
    W = C.EDNET_EVOL_WINDOW
    if n_ans >= 2 * W:
        row["acuracia_inicio"] = round(float(correct[:W].mean()), 4)
        row["acuracia_fim"] = round(float(correct[-W:].mean()), 4)
    fsrc = final["source"].fillna("desconhecido").to_numpy()
    # Evolução sem o diagnóstico: o diagnóstico (nivelamento, mais difícil) ocupa em média 36% das
    # "primeiras 20" questões e quase nada das "últimas 20", o que infla a evolução. Aqui as duas
    # janelas usam apenas questões fora do diagnóstico.
    c_nd = correct[fsrc != "diagnosis"]
    if len(c_nd) >= 2 * W:
        row["acuracia_inicio_sem_diag"] = round(float(c_nd[:W].mean()), 4)
        row["acuracia_fim_sem_diag"] = round(float(c_nd[-W:].mean()), 4)
    for s in set(fsrc):
        m = fsrc == s
        row[f"resp_{s}"] = int(m.sum())
        row[f"acerto_{s}"] = int(correct[m].sum())

    # ---- tempo médio por bloco de questões (enter b -> submit)
    bt = []
    last_enter = None
    for k in range(len(t)):
        if act[k] == "enter" and prefix[k] == "b":
            last_enter = t[k]
        elif act[k] == "submit" and last_enter is not None:
            d = t[k] - last_enter
            if 0 < d <= MAX_CONTENT_MS:
                bt.append(d)
            last_enter = None
    row["tempo_medio_bloco_seg"] = round(float(np.mean(bt)) / 1000, 3) if bt else np.nan
    return row, q


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    names = listar_arquivos()
    print(f"Fonte: {C.EDNET_KT3}  |  gabarito: {C.EDNET_QUESTIONS}", flush=True)
    if limit:
        names = names[:limit]
    print(f"{len(names):,} arquivos de estudantes", flush=True)
    rows, qual = [], Counter()
    t0 = time.time()
    with Pool(processes=max(1, (__import__("os").cpu_count() or 4) - 2), initializer=_init) as pool:
        for i, (row, q) in enumerate(pool.imap_unordered(process_user, names, chunksize=200), 1):
            qual.update(q)
            if row:
                rows.append(row)
            if i % 20000 == 0:
                print(f"  {i:,}/{len(names):,}  ({time.time() - t0:,.0f}s)", flush=True)
    users = pd.DataFrame(rows)
    src_cols = sorted(c for c in users.columns if c.startswith(("src_", "resp_", "acerto_")))
    users[src_cols] = users[src_cols].fillna(0).astype("int64")
    users = users.sort_values("user_id")
    qual["arquivos"] = len(names)
    qual["usuarios_com_dados"] = len(users)
    suffix = f"_sample{limit}" if limit else ""
    users.to_csv(C.PROCESSED / f"ednet_users{suffix}.csv.gz", index=False)
    (C.TAB / f"ednet_qualidade{suffix}.json").write_text(json.dumps(dict(qual), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(qual), indent=2, ensure_ascii=False))
    print(f"OK em {time.time() - t0:,.0f}s — {len(users):,} estudantes, {users.shape[1]} atributos")


if __name__ == "__main__":
    main()
