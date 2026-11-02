"""EdNet-KT3 — Etapas 4 e 5 do CRISP-DM (modelagem/mineração + avaliação) para Q1, Q4, Q5 e Q10.

Entrada: data/processed/ednet_users.csv.gz (1 linha por estudante, gerado por ednet_01_extract.py)
Saída  : outputs/tables/ednet_resultados.json + outputs/figures/ednet/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import (EDNET, INK2, MUTED, NEG, NEUTRAL, POS, dump, engagement_index, fmt_p, groups_test,
                   partial_spearman, q4_figura, q4_tabela, q10_painel, quartiles, save, spearman)

FIG = C.FIG / "ednet"
R = {}
FREQ = {"n_eventos": "Nº de interações", "n_sessoes": "Nº de sessões", "dias_ativos": "Dias ativos"}
ENG = ["n_eventos", "n_sessoes", "dias_ativos", "tempo_estudo_min"]
SOURCES = {
    "sprint": "Prática por parte (sprint)",
    "diagnosis": "Diagnóstico inicial",
    "adaptive_offer": "Recomendação adaptativa",
    "review_quiz": "Quiz de revisão",
    "my_note": "Caderno de erros/anotações",
    "archive": "Questões salvas (arquivo)",
    "tutor": "Tutor",
    "review": "Revisão",
}
CONTENT = {"acessos_explicacao": "Leitura de explicações (feedback)", "acessos_aula": "Aulas (vídeo/texto)"}

u = pd.read_csv(C.PROCESSED / "ednet_users.csv.gz")
for s in SOURCES:  # garante colunas mesmo se alguma origem não existir
    for p in ("src_", "resp_", "acerto_"):
        u[p + s] = u.get(p + s, 0)
extra_src = sorted(c[4:] for c in u.columns if c.startswith("src_") and c[4:] not in SOURCES)
for s in extra_src:
    SOURCES[s] = s
ATIV = {f"src_{k}": v for k, v in SOURCES.items()} | CONTENT

# desempenho
u["acuracia_diag"] = np.where(u["resp_diagnosis"] >= 5, u["acerto_diagnosis"] / u["resp_diagnosis"], np.nan)
pos_n = u["n_respondidas"] - u["resp_diagnosis"]
u["acuracia_pos_diag"] = np.where(pos_n >= C.EDNET_MIN_ANSWERS, (u["n_acertos"] - u["acerto_diagnosis"]) / pos_n.clip(lower=1), np.nan)
# Evolução (Q5): janelas de 20 respostas FORA do diagnóstico. A versão original (qualquer origem) misturava
# o diagnóstico — mais difícil e concentrado no início — com a prática, inflando a evolução. Fica guardada
# como "_com_diag" só para comparação.
u["evolucao_com_diag"] = u["acuracia_fim"] - u["acuracia_inicio"]
u["acuracia_inicio"], u["acuracia_fim"] = u["acuracia_inicio_sem_diag"], u["acuracia_fim_sem_diag"]
u["evolucao"] = u["acuracia_fim"] - u["acuracia_inicio"]
u["evolucao_diag"] = u["acuracia_pos_diag"] - u["acuracia_diag"]
# Desempenho (Q1/Q10): acurácia só nas questões FORA do diagnóstico (≥ 10). A acurácia geral mistura o diagnóstico
# (mais difícil) com a prática, e quem usa pouco tem quase só diagnóstico: mediana de 100% das respostas no 1º
# quartil de interação × 3% no 4º. Isso inflava a correlação da Q1 (0,36). A versão geral fica só para comparação.
perf = u.dropna(subset=["acuracia_pos_diag"]).copy()
perf["quartil_interacoes"] = quartiles(perf["n_eventos"])
QLAB = list(perf["quartil_interacoes"].cat.categories)
evol = u.dropna(subset=["evolucao"]).copy()
evol["quartil_interacoes"] = quartiles(evol["n_eventos"])


def binned(ax, x, y, color, label=None, nb=12):
    """Mediana e intervalo interquartil de y por faixas (quantis) de x — legível com centenas de milhares de pontos."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d["b"] = pd.qcut(np.log10(d["x"]), nb, duplicates="drop")
    g = d.groupby("b", observed=True).agg(xm=("x", "median"), med=("y", "median"),
                                          q1=("y", lambda s: s.quantile(.25)), q3=("y", lambda s: s.quantile(.75)),
                                          mean=("y", "mean"))
    ax.fill_between(g["xm"], g["q1"], g["q3"], color=color, alpha=0.15, lw=0)
    ax.plot(g["xm"], g["mean"], color=color, lw=2, marker="o", ms=5, label=label or "média")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return g


# ============================================================ Descritiva
def descritiva():
    qual = json.loads((C.TAB / "ednet_qualidade.json").read_text(encoding="utf-8"))
    R["descritiva"] = {
        "estudantes": len(u), "eventos_limpos": int(u["n_eventos"].sum()),
        "estudantes_com_10+_respostas_fora_diag": len(perf),
        "estudantes_com_40+_respostas_fora_diag": len(evol),
        "fluxo_q5": {
            "base_original": len(u),
            "com_ao_menos_1_resposta_final": int((u["n_respondidas"] > 0).sum()),
            "com_ao_menos_1_resposta_fora_do_diagnostico": int((pos_n > 0).sum()),
            "com_40+_respostas_qualquer_origem (formula antiga)": int((u["n_respondidas"] >= 2 * C.EDNET_EVOL_WINDOW).sum()),
            "com_40+_respostas_fora_do_diagnostico (amostra final Q5)": int((pos_n >= 2 * C.EDNET_EVOL_WINDOW).sum()),
            "evolucao_calculada": int(u["evolucao"].notna().sum())},
        "estatisticas": u[ENG + ["n_respondidas", "acuracia", "pct_mobile"]].describe().round(3).to_dict(),
        "acuracia_media_geral": round(float(u["n_acertos"].sum() / u["n_respondidas"].sum()), 4),
        "pct_so_uma_sessao": round(float((u["n_sessoes"] == 1).mean() * 100), 1),
        "pct_so_diagnostico": round(float((u["n_eventos"] == u["src_diagnosis"]).mean() * 100), 1),
        "qualidade": qual,
    }
    acts = u[[f"acao_{a}" for a in ("enter", "respond", "submit", "quit")]].sum().sort_values()
    srcs = u[[f"src_{s}" for s in SOURCES]].sum().sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, s, labels, title in ((axs[0], acts, [c[5:] for c in acts.index], "Tipo de ação (action_type)"),
                                 (axs[1], srcs, [SOURCES[c[4:]] for c in srcs.index], "Origem da atividade (source)")):
        ax.barh(labels, s.values / 1e6, color=EDNET, height=0.65)
        for i, v in enumerate(s.values):
            ax.text(v / 1e6, i, f" {v / 1e6:.1f} mi".replace(".", ","), va="center", fontsize=8, color=INK2)
        ax.set_title(title)
        ax.set_xlabel("Milhões de eventos")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, s.max() / 1e6 * 1.2)
    fig.suptitle("EdNet-KT3 — Distribuição das interações", x=0.0, ha="left", fontweight="bold", y=1.03)
    save(fig, FIG / "00_acoes_origens.png",
         f"{int(u['n_eventos'].sum()):,} eventos limpos de {len(u):,} estudantes".replace(",", "."))


# ============================================================ Q1
def q1():
    out = {"populacao": len(perf), "desempenho": "acurácia nas respostas finais fora do diagnóstico (≥10 questões)",
           "correlacoes": {}, "parciais_controle_diagnostico": {}}
    ctrl = u.dropna(subset=["acuracia_diag", "acuracia_pos_diag"])
    for c in FREQ:
        out["correlacoes"][c] = spearman(perf[c], perf["acuracia_pos_diag"])
        out["parciais_controle_diagnostico"][c] = partial_spearman(ctrl[c], ctrl["acuracia_pos_diag"], ctrl["acuracia_diag"])
    out["populacao_parcial"] = len(ctrl)
    out["grupos_quartil_interacoes"] = groups_test(perf, "quartil_interacoes", "acuracia_pos_diag")
    geral = u[u["n_respondidas"] >= C.EDNET_MIN_ANSWERS]
    q_geral = quartiles(geral["n_eventos"])
    out["versao_original_acuracia_geral"] = {
        "n": len(geral), "rho_n_eventos": spearman(geral["n_eventos"], geral["acuracia"], boot=0)["rho"],
        "mediana_pct_diagnostico_por_quartil": (geral["resp_diagnosis"] / geral["n_respondidas"]).groupby(
            q_geral, observed=True).median().round(2).to_dict(),
        "rho_n_eventos_x_pct_diagnostico": spearman(geral["n_eventos"], geral["resp_diagnosis"] / geral["n_respondidas"],
                                                    boot=0)["rho"]}
    R["Q1"] = out

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    binned(ax, perf["n_eventos"], perf["acuracia_pos_diag"] * 100, EDNET)
    ax.set_xlabel("Nº de interações (escala log)")
    ax.set_ylabel("Acurácia (%) — média e faixa interquartil")
    r = out["correlacoes"]["n_eventos"]
    ax.set_title("Interações × desempenho")
    ax.text(0.02, 0.97, f"Spearman ρ = {r['rho']:.2f} ({r['forca']}) · {fmt_p(r['p'])}".replace(".", ",", 1),
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax = axs[1]
    g = perf.groupby("quartil_interacoes", observed=True)["acuracia_pos_diag"].mean() * 100
    ax.bar(range(4), g.values, color=["#f6c3ad", "#f19a75", EDNET, "#b94a1e"], width=0.65)
    for i, v in enumerate(g.values):
        ax.text(i, v + 1, f"{v:.1f}%".replace(".", ","), ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(4), [l.replace(" (", "\n(") for l in QLAB])
    ax.set_ylim(0, 100)
    ax.set_title("Acurácia média por quartil de interação")
    ax.set_ylabel("Acurácia média (%)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("EdNet — Q1: frequência de interação × desempenho", x=0.0, ha="left", fontweight="bold", y=1.03)
    save(fig, FIG / "Q1_interacao_desempenho.png",
         f"n = {len(perf):,} estudantes com ≥ {C.EDNET_MIN_ANSWERS} questões respondidas fora do diagnóstico".replace(",", ".")
         + " · acurácia calculada sem as questões do diagnóstico inicial")


# ============================================================ Q4
def q4():
    # 62% dos estudantes só fizeram o diagnóstico e não voltaram. Com eles, "baixo engajamento" vira
    # sinônimo de "só diagnóstico" e as correlações ficam infladas. A análise principal usa quem
    # efetivamente usou a plataforma (comparável ao ARES); o abandono pós-diagnóstico é reportado à parte.
    so_diag = u["n_eventos"] == u["src_diagnosis"]
    R["Q4_abandono_pos_diagnostico"] = {
        "estudantes_so_diagnostico": int(so_diag.sum()), "pct": round(float(so_diag.mean() * 100), 1),
        "mediana_eventos_so_diag": float(u.loc[so_diag, "n_eventos"].median()),
        "acuracia_diag_media_so_diag": round(float(u.loc[so_diag, "acuracia_diag"].mean()), 4),
        "acuracia_diag_media_continuaram": round(float(u.loc[~so_diag, "acuracia_diag"].mean()), 4),
        "pct_mobile_so_diag": round(float(u.loc[so_diag, "pct_mobile"].mean() * 100), 1),
        "pct_mobile_continuaram": round(float(u.loc[~so_diag, "pct_mobile"].mean() * 100), 1),
    }
    d = u[~so_diag].copy()
    t = q4_tabela(d, list(ATIV), d["n_eventos"], "n_eventos", label=lambda c: ATIV[c])
    mob = partial_spearman(d["pct_mobile"], engagement_index(d, ["n_sessoes", "dias_ativos"]), np.log(d["n_eventos"]))
    t.to_csv(C.TAB / "ednet_Q4_atividades_engajamento.csv", index=False)
    R["Q4"] = {"populacao": len(d),
               "indice_bruto": "média dos percentis de interações, sessões, dias ativos e tempo (contém efeito mecânico)",
               "indice_corrigido": "média dos percentis de sessões e dias ativos; ρ parcial controlando log(nº de interações)",
               "atividades": t.to_dict("records"), "uso_mobile_vs_engajamento_corrigido": mob}
    q4_figura(t, "EdNet — Q4: tipos de interação associados ao engajamento",
              f"n = {len(d):,} estudantes que usaram a plataforma além do diagnóstico".replace(",", ".") +
              " · base grande: quase tudo tem p < 0,001; interprete pela magnitude do ρ",
              FIG / "Q4_atividades_engajamento.png")


# ============================================================ Q5
def q5():
    out = {"populacao": len(evol), "evolucao": f"acurácia nas últimas {C.EDNET_EVOL_WINDOW} − primeiras "
                                               f"{C.EDNET_EVOL_WINDOW} questões fora do diagnóstico (estudantes com ≥ {2 * C.EDNET_EVOL_WINDOW})",
           "correlacoes_evolucao": {}, "parciais_evolucao_controle_inicio": {}, "correlacoes_evolucao_diag": {}}
    ed = u.dropna(subset=["evolucao_diag"])
    for c in FREQ:
        out["correlacoes_evolucao"][c] = spearman(evol[c], evol["evolucao"])
        out["parciais_evolucao_controle_inicio"][c] = partial_spearman(evol[c], evol["evolucao"], evol["acuracia_inicio"])
        out["correlacoes_evolucao_diag"][c] = spearman(ed[c], ed["evolucao_diag"], boot=100)
    out["populacao_evolucao_diag"] = len(ed)
    out["grupos_quartil_interacoes"] = groups_test(evol, "quartil_interacoes", "evolucao")
    out["pct_melhorou_por_quartil"] = (evol.groupby("quartil_interacoes", observed=True)["evolucao"]
                                       .apply(lambda s: round(float((s > 0).mean() * 100), 1)).to_dict())
    out["media_inicio_fim"] = {"inicio": round(float(evol.acuracia_inicio.mean()), 4),
                               "fim": round(float(evol.acuracia_fim.mean()), 4),
                               "pct_melhorou": round(float((evol.evolucao > 0).mean() * 100), 1)}
    ec = u.dropna(subset=["evolucao_com_diag"])
    out["versao_original_com_diagnostico"] = {
        "n": len(ec), "evolucao_media": round(float(ec.evolucao_com_diag.mean()), 4),
        "pct_melhorou": round(float((ec.evolucao_com_diag > 0).mean() * 100), 1),
        "rho_n_eventos": spearman(ec["n_eventos"], ec["evolucao_com_diag"], boot=0)["rho"]}
    out["evolucao_mediana"] = round(float(evol.evolucao.median()), 4)
    out["pct_piorou"] = round(float((evol.evolucao < 0).mean() * 100), 1)
    out["regressao_a_media_rho_inicio_vs_evolucao"] = spearman(evol["acuracia_inicio"], evol["evolucao"], boot=0)["rho"]
    R["Q5"] = out

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    g = evol.groupby("quartil_interacoes", observed=True)[["acuracia_inicio", "acuracia_fim"]].mean() * 100
    for i, (a, b) in enumerate(g.values):
        ax.plot([a, b], [i, i], color=NEUTRAL, lw=2, zorder=1)
        ax.scatter(a, i, s=60, color="white", edgecolor=MUTED, linewidth=1.5, zorder=2)
        ax.scatter(b, i, s=60, color=EDNET, zorder=3)
        ax.text(max(a, b) + 0.6, i, f"{b - a:+.1f} p.p.".replace(".", ",", 1), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(4), QLAB)
    ax.set_xlabel("Acurácia média (%)")
    ax.set_title(f"Primeiras {C.EDNET_EVOL_WINDOW} (○) → últimas {C.EDNET_EVOL_WINDOW} (●) questões*")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(g.values.min() - 3, g.values.max() + 6)
    ax = axs[1]
    binned(ax, evol["n_eventos"], evol["evolucao"] * 100, EDNET)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Nº de interações (escala log)")
    ax.set_ylabel("Evolução da acurácia (p.p.)")
    r = out["correlacoes_evolucao"]["n_eventos"]
    ax.set_title("Interações × evolução")
    ax.text(0.02, 0.97, f"Spearman ρ = {r['rho']:.2f} ({r['forca']}) · {fmt_p(r['p'])}".replace(".", ",", 1),
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    fig.suptitle("EdNet — Q5: frequência de interação × evolução do desempenho", x=0.0, ha="left",
                 fontweight="bold", y=1.03)
    save(fig, FIG / "Q5_interacao_evolucao.png",
         f"n = {len(evol):,} estudantes com ≥ {2 * C.EDNET_EVOL_WINDOW} questões respondidas fora do diagnóstico".replace(",", ".")
         + " · *questões do diagnóstico inicial excluídas das duas janelas")


# ============================================================ Q10
def q10():
    rows = []
    tot = u["n_eventos"].sum()
    ativos = u[u["n_eventos"] > u["src_diagnosis"]]
    ctrl = u.dropna(subset=["acuracia_diag", "acuracia_pos_diag"])
    vol = pd.DataFrame({"diag": ctrl["acuracia_diag"], "log_volume": np.log(ctrl["n_eventos"])})
    for c, lab in ATIV.items():
        s = c[4:] if c.startswith("src_") else None
        r = spearman(perf[c], perf["acuracia_pos_diag"], boot=0)
        rp = partial_spearman(ctrl[c], ctrl["acuracia_pos_diag"], ctrl["acuracia_diag"])
        rv = partial_spearman(ctrl[c], ctrl["acuracia_pos_diag"], vol)
        acc_in = (u[f"acerto_{s}"].sum() / u[f"resp_{s}"].sum()) if s and u[f"resp_{s}"].sum() else np.nan
        rows.append({"atividade": lab, "coluna": c,
                     "tipo": "origem (partição dos eventos)" if s else "acessos (enter; já contados numa origem)",
                     "eventos_total": int(u[c].sum()),
                     "pct_eventos": round(float(u[c].sum() / tot * 100), 2) if s else None,
                     "pct_estudantes_que_usam": round(float((u[c] > 0).mean() * 100), 1),
                     "pct_estudantes_ativos_que_usam": round(float((ativos[c] > 0).mean() * 100), 1),
                     "media_por_estudante_usuario": round(float(u.loc[u[c] > 0, c].mean()), 1),
                     "acuracia_dentro_da_atividade": round(float(acc_in), 4) if acc_in == acc_in else None,
                     "rho_acuracia_pos_diag": r["rho"], "p_acuracia_pos_diag": r["p"],
                     "rho_parcial_acuracia_ctrl_diag": rp["rho_parcial"], "p_parcial": rp["p"],
                     "rho_parcial_ctrl_diag_e_volume": rv["rho_parcial"], "p_parcial_ctrl_volume": rv["p"]})
    t = pd.DataFrame(rows).sort_values("eventos_total", ascending=False)
    t.to_csv(C.TAB / "ednet_Q10_atividades_desempenho.csv", index=False)
    R["Q10"] = {"populacao_uso": len(u), "populacao_ativos": len(ativos), "populacao_desempenho": len(perf),
                "populacao_parcial": len(ctrl),
                "soma_das_origens_igual_total": bool((u[[c for c in ATIV if c.startswith("src_")]].sum(axis=1)
                                                      == u["n_eventos"]).all()),
                "denominadores": {"pct_eventos": f"{int(tot)} eventos de todos os {len(u)} estudantes (só origens)",
                                  "pct_estudantes_que_usam": f"todos os {len(u)} estudantes (inclui os só-diagnóstico)",
                                  "pct_estudantes_ativos_que_usam": f"{len(ativos)} estudantes que foram além do diagnóstico",
                                  "correlacoes": f"{len(ctrl)} estudantes com diagnóstico (>=5) e >=10 questões depois"},
                "atividades": t.to_dict("records")}

    t = t.sort_values("pct_estudantes_ativos_que_usam").reset_index(drop=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ax = axs[0]
    ax.barh(range(len(t)), t["pct_estudantes_ativos_que_usam"], color=EDNET, height=0.65)
    for i, r in t.iterrows():
        unid = "eventos" if r["coluna"].startswith("src_") else "acessos"
        txt = f"{r['pct_estudantes_ativos_que_usam']:.1f}% · {r['eventos_total'] / 1e6:.1f} mi {unid}"
        ax.text(r["pct_estudantes_ativos_que_usam"] + 1, i, txt.replace(".", ","), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(t)), t["atividade"])
    ax.set_xlabel(f"% dos {len(ativos):,} estudantes ativos que usaram a atividade".replace(",", "."))
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 101, 20))
    ax.set_title("Frequência de utilização")
    ax.grid(axis="y", visible=False)
    q10_painel(axs[1], t, "rho_parcial_acuracia_ctrl_diag", "p_parcial", "rho_parcial_ctrl_diag_e_volume",
               "ρ parcial: uso × acurácia pós-diagnóstico (controlando o diagnóstico)")
    fig.suptitle("EdNet — Q10: atividades mais usadas e sua relação com o desempenho", x=0.0, ha="left",
                 fontweight="bold", y=1.03)
    save(fig, FIG / "Q10_atividades_desempenho.png",
         f"ativos = foram além do diagnóstico · desempenho: n = {len(ctrl):,} estudantes com diagnóstico (≥5) e ≥10 ".replace(",", ".")
         + "questões depois dele · explicações e aulas são acessos, já contados dentro das origens")


if __name__ == "__main__":
    descritiva(); q1(); q4(); q5(); q10()
    dump(R, C.TAB / "ednet_resultados.json")
    print("EdNet OK")
