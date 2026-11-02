"""Auditoria das análises (revisão): reproduz o r de Pearson do notebook antigo (Q1), compara as duas fórmulas de
evolução do EdNet (Q5) e confere a consistência entre dados processados, tabelas e figuras.
As correções de Q4 (associação mecânica) e Q10 (volume e múltiplos testes) foram incorporadas às análises principais.

Saída: outputs/tables/auditoria.json (+ impressão resumida)
Não altera nenhum resultado; só recalcula e compara.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from ares_01_prepare import IDS_REMOVER
from utils import dump, partial_spearman, spearman

OUT = {}
EXT = "http://mintly.kibi.group/contextExtension/fragment"


def boot_pearson(x, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    bs = [stats.pearsonr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return [round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])]


def correl(x, y):
    r, pr = stats.pearsonr(x, y)
    s = spearman(x, y, boot=2000)
    return {"n": len(x), "pearson_r": round(float(r), 4), "pearson_p": round(float(pr), 4),
            "pearson_ic95": boot_pearson(x, y), "spearman_rho": s["rho"], "spearman_p": round(s["p"], 4),
            "spearman_ic95": s["ic95"]}


# ======================================================================= Q1
def q1():
    """Reproduz o r = 0,118 do notebook antigo e mede o efeito de cada critério de limpeza."""
    x = pd.read_excel(C.ARES_DIR / "preprocessed" / "xapi_statement_student.xlsx")
    x["fragment"] = x["context"].apply(lambda c: json.loads(c).get("extensions", {}).get(EXT))
    pre_r = pd.read_excel(C.ARES_RAW / "pre_test_response.xlsx")
    pos_r = pd.read_excel(C.ARES_RAW / "post_test_response.xlsx")
    pos_off = pd.read_excel(C.ARES_RAW / "post_test_score.xlsx").set_index("user_id")["score"]
    # nota do notebook antigo: soma das 12 questões (inclui a Q9, que a nota oficial não conta)
    pos12 = pos_r.dropna(subset=["user_id"]).assign(c=lambda d: d.correct.eq("t")).groupby("user_id").c.sum()
    pre12 = pre_r.assign(c=lambda d: d.correct.eq("t")).groupby("user_id").c.sum()
    alunos_antigo = sorted(set(pre12.index) & set(pos12.index) & set(x.actor))

    final = pd.read_csv(C.PROCESSED / "ares_alunos.csv")
    final = final[final.n_interacoes.gt(0)].dropna(subset=["nota_pre", "nota_pos"]).set_index("user_id")
    OUT["Q1_mesmos_alunos"] = {"n_notebook_antigo": len(alunos_antigo), "n_documentacao": len(final),
                               "conjuntos_identicos": set(alunos_antigo) == set(final.index),
                               "removidos_pela_lista_dos_autores_na_intersecao": len(set(alunos_antigo) & IDS_REMOVER)}
    ids = final.index
    todos = x.groupby("actor").size()
    sem_dup = x.drop_duplicates().groupby("actor").size()
    cenarios = {
        "A_notebook_antigo (todos os eventos, com duplicatas, nota 0–12)": (todos, pos12),
        "B = A + nota oficial 0–11 (sem Q9)": (todos, pos_off),
        "C = B + sem duplicatas": (sem_dup, pos_off),
        "D = C + só eventos de aprendizagem (DOCUMENTAÇÃO)": (final["n_interacoes"], pos_off),
    }
    res = {}
    for nome, (inter, nota) in cenarios.items():
        d = pd.DataFrame({"i": inter.reindex(ids), "n": nota.reindex(ids)}).dropna()
        res[nome] = correl(d.i, d.n)
    d = pd.DataFrame({"i": final["n_interacoes"], "n": final["nota_pos"]})
    res["D com log(interações) — Pearson"] = {"pearson_r": round(float(stats.pearsonr(np.log(d.i), d.n)[0]), 4),
                                              "pearson_p": round(float(stats.pearsonr(np.log(d.i), d.n)[1]), 4)}
    sem_out = d[~final["outlier_interacoes"]]
    res["D sem os 4 outliers (IQR)"] = correl(sem_out.i, sem_out.n)
    res["assimetria_interacoes"] = round(float(stats.skew(d.i)), 2)
    res["shapiro_p_interacoes"] = float(stats.shapiro(d.i).pvalue)
    res["n_minimo_para_detectar_r_0.118_com_80pct_poder"] = int(np.ceil(((1.96 + 0.8416) / np.arctanh(0.118)) ** 2 + 3))
    OUT["Q1_decomposicao"] = res


# ======================================================================= Q5
def q5():
    a = pd.read_csv(C.PROCESSED / "ares_alunos.csv").dropna(subset=["nota_pre", "nota_pos"])
    a = a[a.n_interacoes > 0]
    pre_r = pd.read_excel(C.ARES_RAW / "pre_test_response.xlsx")
    q9 = pre_r[pre_r.prompt != "PreTestQ9"].assign(c=lambda d: d.correct.eq("t")).groupby("user_id").c.sum()
    OUT["Q5_ares"] = {
        "formula": "evolucao = nota_pos − nota_pre (0–11, Q9 excluída nas duas provas)",
        "nota_pre_confere_com_recalculo": bool((q9.reindex(a.user_id).values == a.nota_pre.values).all()),
        "ganho_normalizado_indefinido_pre_11": int((a.nota_pre == 11).sum()),
        "ganho_normalizado_min": float(a.ganho_normalizado.min()),
        "ganho_normalizado_pct_negativos": round(float((a.ganho_normalizado < 0).mean() * 100), 1),
        "rho_evolucao_interacoes": spearman(a.n_interacoes, a.evolucao, boot=2000),
        "poder": "n = 141 detecta |ρ| ≥ 0,24 com 80% de poder; o IC95% [−0,17; +0,18] não descarta efeito fraco",
    }
    e = pd.read_csv(C.PROCESSED / "ednet_users.csv.gz")
    if "acuracia_inicio_sem_diag" in e:
        v1 = e.dropna(subset=["acuracia_inicio"])
        v2 = e.dropna(subset=["acuracia_inicio_sem_diag"])
        ev1 = v1.acuracia_fim - v1.acuracia_inicio
        ev2 = v2.acuracia_fim_sem_diag - v2.acuracia_inicio_sem_diag
        OUT["Q5_ednet"] = {
            "formula_original": "últimas 20 − primeiras 20 respostas (qualquer origem)",
            "formula_corrigida": "últimas 20 − primeiras 20 respostas FORA do diagnóstico",
            "original": {"n": len(v1), "inicio": round(float(v1.acuracia_inicio.mean()), 4),
                         "fim": round(float(v1.acuracia_fim.mean()), 4), "evolucao_media": round(float(ev1.mean()), 4),
                         "pct_melhorou": round(float((ev1 > 0).mean() * 100), 1),
                         "rho_n_eventos": spearman(v1.n_eventos, ev1, boot=100)},
            "corrigida": {"n": len(v2), "inicio": round(float(v2.acuracia_inicio_sem_diag.mean()), 4),
                          "fim": round(float(v2.acuracia_fim_sem_diag.mean()), 4),
                          "evolucao_media": round(float(ev2.mean()), 4),
                          "evolucao_mediana": round(float(ev2.median()), 4),
                          "pct_melhorou": round(float((ev2 > 0).mean() * 100), 1),
                          "pct_piorou": round(float((ev2 < 0).mean() * 100), 1),
                          "rho_n_eventos": spearman(v2.n_eventos, ev2, boot=100),
                          "rho_n_sessoes": spearman(v2.n_sessoes, ev2, boot=100),
                          "rho_dias_ativos": spearman(v2.dias_ativos, ev2, boot=100),
                          "parcial_n_eventos_ctrl_inicio": partial_spearman(v2.n_eventos, ev2, v2.acuracia_inicio_sem_diag)},
        }


# ======================================================================= Consistência
def consistencia():
    """Confere se resultados, tabelas e figuras vêm da mesma execução dos dados processados."""
    arquivos = {p.relative_to(C.ROOT).as_posix(): p.stat().st_mtime
                for p in list((C.PROCESSED).glob("*")) + list(C.TAB.glob("*")) + list(C.FIG.rglob("*.png"))}
    ref = {"ares": (C.PROCESSED / "ares_alunos.csv").stat().st_mtime,
           "ednet": (C.PROCESSED / "ednet_users.csv.gz").stat().st_mtime}
    desatualizados = [k for k, t in arquivos.items()
                      if ("ares" in k and "processed" not in k and t < ref["ares"]) or
                         ("ednet" in k and "processed" not in k and "qualidade" not in k and "extract" not in k and t < ref["ednet"]) or
                         ("comparacao" in k and t < max(ref.values()))]
    OUT["consistencia"] = {"saidas_mais_antigas_que_os_dados_processados": desatualizados,
                           "recalculo_independente": recalculo_independente()}


# ---- Recálculo independente: NÃO usa utils.py; reimplementa filtros e fórmulas a partir dos dados processados e
# compara com os JSONs gerados pelas análises. Uma divergência indica erro de código ou resultado desatualizado.
def _rank_resid(y, Z):
    y = pd.Series(y).rank().to_numpy()
    Z = np.column_stack([np.ones(len(y))] + [pd.Series(z).rank().to_numpy() for z in Z])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def _parcial(x, y, *z):
    d = pd.concat([pd.Series(np.asarray(x)), pd.Series(np.asarray(y))] + [pd.Series(np.asarray(v)) for v in z],
                  axis=1).dropna()
    zs = [d[i] for i in range(2, d.shape[1])]
    return float(np.corrcoef(_rank_resid(d[0], zs), _rank_resid(d[1], zs))[0, 1])


def _holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    adj[o] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    return adj


def recalculo_independente():
    A = json.loads((C.TAB / "ares_resultados.json").read_text(encoding="utf-8"))
    E = json.loads((C.TAB / "ednet_resultados.json").read_text(encoding="utf-8"))
    checks = []

    def ok(nome, oficial, recalculado, tol=1e-3):
        checks.append({"verificacao": nome, "oficial": oficial, "recalculado": round(float(recalculado), 4),
                       "confere": bool(abs(float(oficial) - float(recalculado)) <= tol)})

    # ---------- ARES
    a = pd.read_csv(C.PROCESSED / "ares_alunos.csv")
    log = a[a.n_interacoes > 0]
    perf = log.dropna(subset=["nota_pre", "nota_pos"])
    ok("ARES Q1 n", A["Q1"]["populacao"], len(perf), 0)
    ok("ARES Q1 ρ interações×nota pós", A["Q1"]["correlacoes"]["n_interacoes"]["rho"],
       stats.spearmanr(perf.n_interacoes, perf.nota_pos)[0])
    ok("ARES Q1 ρ parcial (pré)", A["Q1"]["parciais_controle_pre"]["n_interacoes"]["rho_parcial"],
       _parcial(perf.n_interacoes, perf.nota_pos, perf.nota_pre))
    ok("ARES Q5 ρ interações×evolução", A["Q5"]["correlacoes_evolucao"]["n_interacoes"]["rho"],
       stats.spearmanr(perf.n_interacoes, perf.nota_pos - perf.nota_pre)[0])
    ativ = [c for c in a.columns if c.startswith("ativ::") and "Alertas" not in c]
    persist = log[["n_sessoes", "dias_ativos"]].rank(pct=True).mean(axis=1)
    tot = log[ativ].sum(axis=1)
    q4 = {r["atividade"]: r for r in A["Q4"]["atividades"]}
    p4 = []
    for c in ativ:
        r = _parcial(log[c] / tot, persist, np.log(log.n_interacoes))
        ok(f"ARES Q4 ρ corrigido — {c[6:]}", q4[c[6:]]["rho_corrigido"], r)
        n = len(log)
        p4.append(2 * stats.t.sf(abs(r) * np.sqrt((n - 3) / (1 - r * r)), n - 3))  # gl = n − 2 − 1 controle
    for c, ph in zip(ativ, _holm(p4)):
        ok(f"ARES Q4 p Holm — {c[6:]}", q4[c[6:]]["p_corrigido_holm"], ph, 5e-3)
    q10 = {r["atividade"]: r for r in A["Q10"]["atividades"]}
    p10 = []
    for c in ativ:
        r = _parcial(perf[c], perf.nota_pos, perf.nota_pre)
        ok(f"ARES Q10 ρ parcial (pré) — {c[6:]}", q10[c[6:]]["rho_parcial_nota_pos_ctrl_pre"], r)
        n = len(perf)
        p10.append(2 * stats.t.sf(abs(r) * np.sqrt((n - 3) / (1 - r * r)), n - 3))
        ok(f"ARES Q10 % eventos — {c[6:]}", q10[c[6:]]["pct_eventos"], log[c].sum() / log[ativ].sum().sum() * 100, 0.01)
    for c, ph in zip(ativ, _holm(p10)):
        ok(f"ARES Q10 p Holm — {c[6:]}", q10[c[6:]]["p_parcial_holm"], ph, 5e-3)

    # ---------- EdNet
    e = pd.read_csv(C.PROCESSED / "ednet_users.csv.gz")
    nd = e.n_respondidas - e.resp_diagnosis
    acc_nd = np.where(nd >= C.EDNET_MIN_ANSWERS, (e.n_acertos - e.acerto_diagnosis) / nd.clip(lower=1), np.nan)
    acc_dg = np.where(e.resp_diagnosis >= 5, e.acerto_diagnosis / e.resp_diagnosis.clip(lower=1), np.nan)
    m = ~np.isnan(acc_nd)
    ok("EdNet Q1 n", E["Q1"]["populacao"], m.sum(), 0)
    ok("EdNet Q1 ρ interações×acurácia fora do diag", E["Q1"]["correlacoes"]["n_eventos"]["rho"],
       stats.spearmanr(e.n_eventos[m], acc_nd[m])[0])
    mc = m & ~np.isnan(acc_dg)
    ok("EdNet Q1 ρ parcial (diagnóstico)", E["Q1"]["parciais_controle_diagnostico"]["n_eventos"]["rho_parcial"],
       _parcial(e.n_eventos[mc], acc_nd[mc], acc_dg[mc]))
    ev = e.acuracia_fim_sem_diag - e.acuracia_inicio_sem_diag
    me = ev.notna()
    ok("EdNet Q5 n", E["Q5"]["populacao"], me.sum(), 0)
    ok("EdNet Q5 n = alunos com ≥40 respostas fora do diag", E["Q5"]["populacao"], (nd >= 2 * C.EDNET_EVOL_WINDOW).sum(), 0)
    ok("EdNet Q5 acurácia início", E["Q5"]["media_inicio_fim"]["inicio"], e.acuracia_inicio_sem_diag[me].mean(), 1e-4)
    ok("EdNet Q5 acurácia fim", E["Q5"]["media_inicio_fim"]["fim"], e.acuracia_fim_sem_diag[me].mean(), 1e-4)
    ok("EdNet Q5 ρ interações×evolução", E["Q5"]["correlacoes_evolucao"]["n_eventos"]["rho"],
       stats.spearmanr(e.n_eventos[me], ev[me])[0])
    ativos = e[e.n_eventos > e.src_diagnosis]
    persist = ativos[["n_sessoes", "dias_ativos"]].rank(pct=True).mean(axis=1)
    e4 = {r["coluna"]: r for r in E["Q4"]["atividades"]}
    for c, r in e4.items():
        ok(f"EdNet Q4 ρ corrigido — {r['atividade']}", r["rho_corrigido"],
           _parcial(ativos[c] / ativos.n_eventos, persist, np.log(ativos.n_eventos)))
    src = [c for c in e.columns if c.startswith("src_")]
    ok("EdNet Q10 origens somam o total de eventos", 0, int((e[src].sum(axis=1) != e.n_eventos).sum()), 0)
    e10 = {r["coluna"]: r for r in E["Q10"]["atividades"]}
    for c, r in e10.items():
        ok(f"EdNet Q10 % ativos que usam — {r['atividade']}", r["pct_estudantes_ativos_que_usam"],
           (ativos[c] > 0).mean() * 100, 0.06)
        ok(f"EdNet Q10 ρ parcial (diag + volume) — {r['atividade']}", r["rho_parcial_ctrl_diag_e_volume"],
           _parcial(e[c][mc], acc_nd[mc], acc_dg[mc], np.log(e.n_eventos[mc])))
    falhas = [c for c in checks if not c["confere"]]
    return {"total": len(checks), "conferem": len(checks) - len(falhas), "falhas": falhas, "detalhe": checks}


# ======================================================================= Documento × resultados
def documentacao():
    """Formata os números-chave a partir dos JSONs oficiais e confere se aparecem no documento/README.
    Pega texto desatualizado depois de uma reexecução que mude resultados."""
    A = json.loads((C.TAB / "ares_resultados.json").read_text(encoding="utf-8"))
    E = json.loads((C.TAB / "ednet_resultados.json").read_text(encoding="utf-8"))
    doc = (C.ROOT / "docs" / "DOCUMENTACAO.md").read_text(encoding="utf-8")
    readme = (C.ROOT / "README.md").read_text(encoding="utf-8")
    sg = lambda v: f"{v:+.2f}".replace(".", ",").replace("-", "−")
    pc = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")
    milhar = lambda v: f"{v:,}".replace(",", ".")
    a4 = {r["atividade"]: r for r in A["Q4"]["atividades"]}
    e4 = {r["atividade"]: r for r in E["Q4"]["atividades"]}
    a10 = {r["atividade"]: r for r in A["Q10"]["atividades"]}
    e10 = {r["atividade"]: r for r in E["Q10"]["atividades"]}
    q5 = E["Q5"]
    esperado = {
        "ARES Q1 ρ": sg(A["Q1"]["correlacoes"]["n_interacoes"]["rho"]),
        "EdNet Q1 n": milhar(E["Q1"]["populacao"]),
        "EdNet Q1 ρ": sg(E["Q1"]["correlacoes"]["n_eventos"]["rho"]),
        "EdNet Q1 ρ parcial": sg(E["Q1"]["parciais_controle_diagnostico"]["n_eventos"]["rho_parcial"]),
        "ARES Q4 leitura": sg(a4["Leitura de textos"]["rho_corrigido"]),
        "ARES Q4 navegação": sg(a4["Navegação entre páginas"]["rho_corrigido"]),
        "ARES Q4 feedback": sg(a4["Visualização de feedback/nota"]["rho_corrigido"]),
        "ARES Q4 Holm turmas": pc(a4["Turmas (entrar/sair)"]["p_corrigido_holm"], 2),
        "EdNet Q4 aulas": sg(e4["Aulas (vídeo/texto)"]["rho_corrigido"]),
        "EdNet Q4 explicações": sg(e4["Leitura de explicações (feedback)"]["rho_corrigido"]),
        "EdNet Q5 n": milhar(q5["populacao"]),
        "EdNet Q5 início": pc(q5["media_inicio_fim"]["inicio"] * 100) + "%",
        "EdNet Q5 fim": pc(q5["media_inicio_fim"]["fim"] * 100) + "%",
        "EdNet Q5 ρ": sg(q5["correlacoes_evolucao"]["n_eventos"]["rho"]),
        "EdNet Q5 fluxo ≥1 fora do diag": milhar(E["descritiva"]["fluxo_q5"]["com_ao_menos_1_resposta_fora_do_diagnostico"]),
        "ARES Q10 Holm tutorial": pc(a10["Tutorial"]["p_parcial_holm"], 3),
        "ARES Q10 Holm vocabulário": pc(a10["Vocabulário (palavras e traduções)"]["p_parcial_holm"], 2),
        "EdNet Q10 prática": sg(e10["Prática por parte (sprint)"]["rho_parcial_acuracia_ctrl_diag"]),
        "EdNet Q10 prática + volume": sg(e10["Prática por parte (sprint)"]["rho_parcial_ctrl_diag_e_volume"]),
        "EdNet Q10 explicações + volume": sg(e10["Leitura de explicações (feedback)"]["rho_parcial_ctrl_diag_e_volume"]),
        "EdNet Q10 prática % ativos": pc(e10["Prática por parte (sprint)"]["pct_estudantes_ativos_que_usam"]) + "%",
    }
    no_readme = {"EdNet Q1 ρ", "ARES Q4 leitura", "EdNet Q4 aulas", "EdNet Q5 ρ", "ARES Q10 Holm tutorial"}
    res = []
    for k, v in esperado.items():
        res.append({"valor": k, "esperado": v, "no_documento": v in doc,
                    "no_readme": (v.replace("−", "-") in readme or v in readme) if k in no_readme else None})
    OUT["documentacao"] = {"verificacoes": res,
                           "faltando": [r for r in res if not r["no_documento"] or r["no_readme"] is False]}


if __name__ == "__main__":
    etapas = sys.argv[1:] or ["q1", "q5", "consistencia", "documentacao"]
    for nome in etapas:
        globals()[nome]()
    dump(OUT, C.TAB / "auditoria.json")
    if "consistencia" in OUT:
        ri = OUT["consistencia"]["recalculo_independente"]
        print(f"Recálculo independente: {ri['conferem']}/{ri['total']} verificações conferem.")
        for f in ri["falhas"]:
            print(f"  DIVERGE: {f['verificacao']}: oficial {f['oficial']} × recalculado {f['recalculado']}")
        print("Saídas mais antigas que os dados processados:",
              OUT["consistencia"]["saidas_mais_antigas_que_os_dados_processados"] or "nenhuma")
    falta_doc = OUT.get("documentacao", {}).get("faltando", [])
    if "documentacao" in OUT:
        n = len(OUT["documentacao"]["verificacoes"])
        print(f"Documento × resultados: {n - len(falta_doc)}/{n} números-chave encontrados no texto.")
        for f in falta_doc:
            print(f"  FALTA: {f['valor']} = {f['esperado']} (documento: {f['no_documento']}, README: {f['no_readme']})")
    if "consistencia" in OUT and (OUT["consistencia"]["recalculo_independente"]["falhas"]
                                  or OUT["consistencia"]["saidas_mais_antigas_que_os_dados_processados"]) or falta_doc:
        sys.exit(1)
