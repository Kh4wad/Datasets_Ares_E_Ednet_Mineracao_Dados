"""ARES — Etapas 2 e 3 do CRISP-DM (entendimento + preparação dos dados).

Entradas : data/raw/ares/data/{raw,preprocessed}/*.xlsx (OSF 8y3zp)
Saídas   : data/processed/ares_eventos.csv.gz   (eventos limpos e categorizados)
           data/processed/ares_alunos.csv        (1 linha por aluno: notas + atributos de interação)
           outputs/tables/ares_qualidade.json    (registro de cada passo de limpeza/validação)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import dump

Q = {}  # relatório de qualidade
EXT = "http://mintly.kibi.group/contextExtension/"

# Contas a remover — definidas pelos autores do dataset em script/02_data_user_removal.ipynb
REMOVER = {
    "conta_teste": ["EKEi"],
    "professor": ["82PN", "0jWs", "W2Fl", "FUs4", "VNzA", "P5ze", "OVhL"],
    "conta_nao_usada": ["ZznH", "eVKy", "zRSE", "ImNX", "kxxe", "am1M", "x5ot", "Fl9u", "9FEW", "wU4m", "JmnJ", "ow1W", "Yp1o", "48jO"],
    "sem_submissao": ["eKlW", "WUkv", "zkmh", "m25O", "F1aL", "wgEB", "5AAK", "XfM3", "R49T", "NBrF", "zZSe", "ggi5", "wAr0", "wRnG", "keoO"],
    "sem_consentimento": ["gxjp", "K4b1", "ntzG", "aNxJ", "lOkr", "QUU6", "ewww", "QWJu", "nC8b", "YD86", "s3v8", "g1gg"],
}
IDS_REMOVER = {u for v in REMOVER.values() for u in v}

PAGINAS_AVALIACAO = {"pre_test.html", "post_test.html"}
PAGINAS_QUESTIONARIO = {"background_questionnaire.html", "mslq_questionnaire.html", "motivation_questionnaire.html",
                        "perception_questionnaire_en.html", "engagement_questionnaire.html", "consent_students.html"}


def perfil(df, nome):
    return {"arquivo": nome, "linhas": int(len(df)), "colunas": list(df.columns),
            "nulos": {c: int(v) for c, v in df.isna().sum().items()}, "duplicadas": int(df.duplicated().sum())}


# ------------------------------------------------------------------ notas
def carregar_notas():
    out = {}
    for t, nome in (("pre", "pre_test"), ("post", "post_test")):
        resp = pd.read_excel(C.ARES_RAW / f"{nome}_response.xlsx")
        score = pd.read_excel(C.ARES_RAW / f"{nome}_score.xlsx")
        Q[f"{t}_resposta_bruto"] = perfil(resp, f"raw/{nome}_response.xlsx")
        Q[f"{t}_nota_bruto"] = perfil(score, f"raw/{nome}_score.xlsx")

        # sanitização
        n0 = len(resp)
        resp = resp.dropna(subset=["user_id"]).drop_duplicates()
        resp["user_id"] = resp["user_id"].str.strip()
        sem_resposta = int(resp["response"].isna().sum())
        resp["correct"] = resp["correct"].str.strip().str.lower()
        invalid_correct = int((~resp["correct"].isin(["t", "f"])).sum())
        resp["acertou"] = resp["correct"].eq("t")
        score["user_id"] = score["user_id"].str.strip()
        dup_ids = int(score["user_id"].duplicated().sum())
        score = score.drop_duplicates("user_id")
        fora_faixa = int((~score["score"].between(0, 11)).sum())

        # validação cruzada: a nota oficial = soma de acertos das 12 questões, exceto Q9
        q9 = f"{nome.split('_')[0].capitalize()}TestQ9"
        recalc = resp[resp["prompt"] != q9].groupby("user_id")["acertou"].sum()
        comp = score.set_index("user_id")["score"].to_frame().join(recalc.rename("recalc"), how="left")
        bate = float((comp["score"] == comp["recalc"]).mean())
        questoes_por_aluno = resp.groupby("user_id").size().value_counts().to_dict()

        antes = len(score)
        score = score[~score["user_id"].isin(IDS_REMOVER)]
        Q[f"{t}_limpeza"] = {
            "linhas_resposta_sem_user_id_removidas": n0 - len(resp),
            "respostas_em_branco_(contam_como_erro)": sem_resposta,
            "valores_invalidos_em_correct": invalid_correct,
            "user_id_duplicado_na_nota": dup_ids,
            "notas_fora_de_0_a_11": fora_faixa,
            "questoes_por_aluno": questoes_por_aluno,
            "regra_nota_validada": f"nota = acertos nas 12 questões exceto {q9}",
            "taxa_concordancia_nota_recalculada": bate,
            "alunos_removidos_lista_autores": antes - len(score),
            "alunos_validos": int(len(score)),
            "acerto_por_questao": resp.groupby("prompt")["acertou"].mean().round(3).to_dict(),
        }
        out[t] = score.rename(columns={"score": {"pre": "nota_pre", "post": "nota_pos"}[t]})
    notas = out["pre"].merge(out["post"], on="user_id", how="outer")
    mot = pd.read_excel(C.ARES_RAW / "pre_motivation_score.xlsx")
    Q["motivacao_bruto"] = perfil(mot, "raw/pre_motivation_score.xlsx")
    mot = mot.drop_duplicates("user_id")
    mot = mot[~mot["user_id"].isin(IDS_REMOVER)]
    notas = notas.merge(mot, on="user_id", how="left")
    return notas


# ------------------------------------------------------------------ eventos
def categoria(r):
    """Classifica cada evento xAPI em um tipo de atividade com significado pedagógico."""
    v, o, b, d, f = r.verb, r.object, r.btn or "", r.dlg or "", r.fragment or ""
    if d.startswith("Dialog: Error") or d in {"submission conflict dialog", "test already taken modal",
                                              "test/quetionnaire not taken modal"}:
        return "Alertas/erros do sistema"
    if v == "logged-in":
        return "Acesso (login)"
    if o in {"token", "word-explanation"} or "word" in b or "Chinese" in b or b.startswith("Collapse click") \
            or d in {"token popover", "Explain word modal"} or d.startswith("Collapse:") or o == "construct" \
            or d == "construct offcanvas":
        return "Vocabulário (palavras e traduções)"
    if o == "evaluation" or b in {"Button to see evaluation", "Button to see own totalScore",
                                  "Button to see average class totalScore", "Button to see correct answer"}:
        return "Visualização de feedback/nota"
    if b == "star ratings" or d == "Rating alert":
        return "Avaliação do texto (estrelas)"
    if o in {"klass"} or "class" in b.lower() or "klass" in b or "Class" in d:
        return "Turmas (entrar/sair)"
    if "tutorial" in b or d == "tutorial modal":
        return "Tutorial"
    if v == "submitted" or o == "txtView" or "assignment" in b or b in {"Button to show questions",
                                                                         "Button to confirm empty answer submission",
                                                                         "Button to dismiss empty answer submission"} \
            or d in {"Modal for assignment submission", "Submission alert", "Modal for empty answer submission"}:
        return "Responder questões/tarefas"
    if (v == "entered" and f == "text.html") or b in {"button to redirect to text reading page", "Button to exit a text"}:
        return "Leitura de textos"
    if v == "entered" or o == "filter" or "redirect" in b:
        return "Navegação entre páginas"
    return "Outros (diálogos/botões)"


def carregar_eventos(alunos_validos):
    x = pd.read_excel(C.ARES_DIR / "preprocessed" / "xapi_statement_student.xlsx")
    raw = pd.read_excel(C.ARES_RAW / "xapi_statement_student.xlsx", usecols=["id", "is_voided"])
    Q["xapi_bruto"] = {"arquivo": "raw/xapi_statement_student.xlsx", "linhas": int(len(raw)),
                       "statements_anulados_(is_voided=t)": int((raw.is_voided == "t").sum())}
    Q["xapi_preprocessado"] = perfil(x, "preprocessed/xapi_statement_student.xlsx")
    passos = []
    n = len(x)
    x = x.drop_duplicates()
    passos.append(("remover duplicadas exatas", n - len(x), len(x)))

    n = len(x)
    x = x.dropna(subset=["actor", "verb", "timestamp"])
    passos.append(("remover nulos em actor/verb/timestamp", n - len(x), len(x)))

    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, format="mixed", errors="coerce")
    n = len(x)
    x = x[x["timestamp"].between(pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2025-03-31", tz="UTC"))]
    passos.append(("remover timestamp inválido/fora do período do estudo", n - len(x), len(x)))

    def parse(c):
        try:
            return {k.replace(EXT, ""): v for k, v in json.loads(c).get("extensions", {}).items()}
        except (TypeError, ValueError):
            return None
    ctx = x["context"].apply(parse)
    n = len(x)
    x = x[ctx.notna()]
    ctx = ctx[ctx.notna()]
    passos.append(("remover contexto JSON inválido", n - len(x), len(x)))
    x = x.assign(fragment=ctx.str.get("fragment"), btn=ctx.str.get("btnDescription"),
                 dlg=ctx.str.get("dialogDescription"), assignment=ctx.str.get("assignmentId"))

    nao_alunos = x.loc[~x["actor"].isin(alunos_validos), "actor"].value_counts()
    n = len(x)
    x = x[x["actor"].isin(alunos_validos)]
    passos.append(("remover atores que não são alunos válidos (professores, testes, admin, excluídos)", n - len(x), len(x)))

    n = len(x)
    x = x[~x["fragment"].fillna("").str.startswith(("teacher_", "admin_"))]
    passos.append(("remover eventos em páginas de professor/admin", n - len(x), len(x)))

    x["contexto"] = np.select(
        [x["fragment"].isin(PAGINAS_AVALIACAO), x["fragment"].isin(PAGINAS_QUESTIONARIO)],
        ["avaliacao (pré/pós-teste)", "questionario"], "aprendizagem")
    x["atividade"] = x.apply(categoria, axis=1)
    x = x.sort_values(["actor", "timestamp"]).reset_index(drop=True)
    Q["xapi_limpeza"] = [{"passo": p, "removidas": int(r), "restantes": int(k)} for p, r, k in passos]
    Q["xapi_atores_removidos"] = {str(k): int(v) for k, v in nao_alunos.head(30).items()}
    Q["xapi_atores_removidos_total"] = int(len(nao_alunos))
    Q["xapi_eventos_por_contexto"] = x["contexto"].value_counts().to_dict()
    Q["xapi_verbos_alunos"] = x["verb"].value_counts().to_dict()
    return x


def atributos(ev):
    """1 linha por aluno, usando apenas eventos de APRENDIZAGEM (sem pré/pós-teste e questionários)."""
    a = ev[(ev["contexto"] == "aprendizagem") & (ev["atividade"] != "Alertas/erros do sistema")].copy()
    g = a.groupby("actor")
    gap = g["timestamp"].diff().dt.total_seconds()
    a["nova_sessao"] = gap.isna() | (gap > C.SESSION_GAP_MIN * 60)
    a["gap_s"] = gap.where(gap <= C.SESSION_GAP_MIN * 60, 0).fillna(0)
    feat = pd.DataFrame({
        "n_interacoes": g.size(),
        "n_sessoes": a.groupby("actor")["nova_sessao"].sum(),
        "dias_ativos": g["timestamp"].apply(lambda s: s.dt.date.nunique()),
        "tempo_estudo_min": a.groupby("actor")["gap_s"].sum() / 60,
        "n_logins": g["verb"].apply(lambda s: (s == "logged-in").sum()),
        "tarefas_submetidas": g["verb"].apply(lambda s: (s == "submitted").sum()),
        "tarefas_distintas": g["assignment"].nunique(),
        "periodo_dias": (g["timestamp"].max() - g["timestamp"].min()).dt.total_seconds() / 86400,
    })
    ativ = a.pivot_table(index="actor", columns="atividade", values="verb", aggfunc="size", fill_value=0)
    ativ.columns = [f"ativ::{c}" for c in ativ.columns]
    verb = a.pivot_table(index="actor", columns="verb", values="object", aggfunc="size", fill_value=0)
    verb.columns = [f"verbo::{c}" for c in verb.columns]
    return feat.join(ativ).join(verb).reset_index().rename(columns={"actor": "user_id"})


def main():
    notas = carregar_notas()
    alunos_validos = set(notas["user_id"])
    ev = carregar_eventos(alunos_validos)
    feat = atributos(ev)
    df = notas.merge(feat, on="user_id", how="left")
    cnt = [c for c in df.columns if c.startswith(("ativ::", "verbo::"))] + \
          ["n_interacoes", "n_sessoes", "dias_ativos", "tempo_estudo_min", "n_logins", "tarefas_submetidas", "tarefas_distintas"]
    df["tem_log"] = df["n_interacoes"].notna()
    df[cnt] = df[cnt].fillna(0)
    df["evolucao"] = df["nota_pos"] - df["nota_pre"]
    # ganho normalizado de Hake: quanto do espaço possível de melhora foi conquistado
    df["ganho_normalizado"] = np.where(df["nota_pre"] < 11, df["evolucao"] / (11 - df["nota_pre"]), np.nan)

    # outliers (IQR) — sinalizados, não removidos (as análises usam estatística por postos, robusta)
    q1, q3 = df["n_interacoes"].quantile([0.25, 0.75])
    df["outlier_interacoes"] = df["n_interacoes"] > q3 + 1.5 * (q3 - q1)

    Q["base_final"] = {
        "alunos_total": int(len(df)),
        "com_pre": int(df["nota_pre"].notna().sum()), "com_pos": int(df["nota_pos"].notna().sum()),
        "com_pre_e_pos": int(df[["nota_pre", "nota_pos"]].notna().all(axis=1).sum()),
        "com_log_de_interacao": int(df["tem_log"].sum()),
        "com_pre_pos_e_log": int((df[["nota_pre", "nota_pos"]].notna().all(axis=1) & df["tem_log"] & (df["n_interacoes"] > 0)).sum()),
        "outliers_interacoes_IQR": int(df["outlier_interacoes"].sum()),
    }
    df.to_csv(C.PROCESSED / "ares_alunos.csv", index=False)
    ev.drop(columns=["context"]).to_csv(C.PROCESSED / "ares_eventos.csv.gz", index=False)
    dump(Q, C.TAB / "ares_qualidade.json")
    print(json.dumps(Q["base_final"], indent=2), "\n", json.dumps(Q["xapi_limpeza"], indent=2, ensure_ascii=False))
    print(ev["atividade"].value_counts().to_string())


if __name__ == "__main__":
    main()
