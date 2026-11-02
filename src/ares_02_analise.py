"""ARES — Etapas 4 e 5 do CRISP-DM (modelagem/mineração + avaliação) para Q1, Q4, Q5 e Q10.

Entrada: data/processed/ares_alunos.csv, ares_eventos.csv.gz
Saída  : outputs/tables/ares_resultados.json + outputs/figures/ares/*.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import (ARES, INK2, MUTED, NEG, NEUTRAL, POS, SEQ, ajuste_multiplo, dump, engagement_index, fmt_p,
                   groups_test, partial_spearman, q4_figura, q4_tabela, q10_painel, quartiles, save, spearman)

FIG = C.FIG / "ares"
R = {}
FREQ = {"n_interacoes": "Nº de interações", "n_sessoes": "Nº de sessões", "dias_ativos": "Dias ativos",
        "n_logins": "Nº de acessos (logins)"}
ENG = ["n_interacoes", "n_sessoes", "dias_ativos", "tempo_estudo_min"]
SKIP_ATIV = {"Alertas/erros do sistema"}

df = pd.read_csv(C.PROCESSED / "ares_alunos.csv")
ev = pd.read_csv(C.PROCESSED / "ares_eventos.csv.gz", parse_dates=["timestamp"], dtype={"assignment": str})
ATIV = [c for c in df.columns if c.startswith("ativ::") and c[6:] not in SKIP_ATIV]
log = df[df["n_interacoes"] > 0].copy()                                  # Q4: todos com log
perf = log.dropna(subset=["nota_pre", "nota_pos"]).copy()              # Q1/Q5/Q10: pré + pós + log
perf["quartil_interacoes"] = quartiles(perf["n_interacoes"])
QLAB = list(perf["quartil_interacoes"].cat.categories)


# ============================================================ Descritiva
def descritiva():
    d = {"alunos_com_log": len(log), "alunos_analise_desempenho": len(perf),
         "eventos_aprendizagem": int(log["n_interacoes"].sum()),
         "estatisticas": log[ENG + ["n_logins"]].describe().round(2).to_dict(),
         "notas": perf[["nota_pre", "nota_pos", "evolucao", "ganho_normalizado"]].describe().round(3).to_dict(),
         "pct_melhorou": round(float((perf["evolucao"] > 0).mean() * 100), 1),
         "pct_igual": round(float((perf["evolucao"] == 0).mean() * 100), 1),
         "pct_piorou": round(float((perf["evolucao"] < 0).mean() * 100), 1)}
    from scipy.stats import wilcoxon
    w = wilcoxon(perf["nota_pos"], perf["nota_pre"])
    d["wilcoxon_pos_vs_pre_p"] = float(w.pvalue)
    R["descritiva"] = d

    # verbos (15 tipos — base completa de alunos, como no gráfico original)
    vc = ev["verb"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(vc.index, vc.values, color=ARES, height=0.7)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,}".replace(",", "."), va="center", fontsize=8, color=INK2)
    ax.set_title("ARES — Distribuição dos tipos de interação (verbos xAPI)")
    ax.set_xlabel("Quantidade de eventos (alunos válidos, após limpeza)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, vc.max() * 1.15)
    save(fig, FIG / "00_verbos.png", f"{len(ev):,} eventos de {ev['actor'].nunique()} alunos".replace(",", "."))

    # pré vs pós
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, 12.5, 1)
    ax.hist(perf["nota_pre"], bins=bins, color=NEUTRAL, label="Pré-teste", rwidth=0.85)
    ax.hist(perf["nota_pos"], bins=bins, color=ARES, alpha=0.75, label="Pós-teste", rwidth=0.55)
    ax.set_title("ARES — Notas no pré e no pós-teste")
    ax.set_xlabel("Nota (0–11)")
    ax.set_ylabel("Nº de alunos")
    ax.legend()
    ax.set_xticks(range(12))
    save(fig, FIG / "00_notas_pre_pos.png",
         f"n = {len(perf)} alunos com os dois testes · média pré {perf.nota_pre.mean():.2f} → pós {perf.nota_pos.mean():.2f}"
         .replace(".", ","))


# ============================================================ Q1
def q1():
    out = {"populacao": len(perf), "desempenho": "nota no pós-teste (0–11)", "correlacoes": {}, "parciais_controle_pre": {}}
    for c, lab in FREQ.items():
        out["correlacoes"][c] = spearman(perf[c], perf["nota_pos"])
        out["parciais_controle_pre"][c] = partial_spearman(perf[c], perf["nota_pos"], perf["nota_pre"])
    out["grupos_quartil_interacoes"] = groups_test(perf, "quartil_interacoes", "nota_pos")
    R["Q1"] = out

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    rng = np.random.default_rng(1)
    ax.scatter(perf["n_interacoes"], perf["nota_pos"] + rng.uniform(-0.2, 0.2, len(perf)), s=22, color=ARES,
               alpha=0.65, edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Nº de interações de aprendizagem (escala log)")
    ax.set_ylabel("Nota no pós-teste")
    r = out["correlacoes"]["n_interacoes"]
    ax.set_title("Interações × desempenho")
    ax.text(0.02, 0.97, f"Spearman ρ = {r['rho']:.2f} ({r['forca']}) · {fmt_p(r['p'])}".replace(".", ",", 1),
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax = axs[1]
    g = perf.groupby("quartil_interacoes", observed=True)["nota_pos"]
    m, se = g.mean(), g.sem()
    ax.bar(range(4), m.values, color=[SEQ[2], SEQ[3], SEQ[4], SEQ[5]], width=0.65)
    ax.errorbar(range(4), m.values, yerr=1.96 * se.values, fmt="none", ecolor=INK2, capsize=3, lw=1)
    for i, v in enumerate(m.values):
        ax.text(i, 0.25, f"{v:.1f}".replace(".", ","), ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4), [l.replace(" ", "\n") for l in QLAB])
    ax.set_title("Nota média por quartil de interação")
    ax.set_ylabel("Nota média no pós-teste (IC95%)")
    ax.grid(axis="x", visible=False)
    fig.suptitle("ARES — Q1: frequência de interação × desempenho", x=0.0, ha="left", fontweight="bold", y=1.03)
    save(fig, FIG / "Q1_interacao_desempenho.png",
         f"n = {len(perf)} alunos · quartis por nº de interações · Kruskal-Wallis {fmt_p(out['grupos_quartil_interacoes']['kruskal_p'])}")


# ============================================================ Q4
def q4():
    t = q4_tabela(log, ATIV, log[ATIV].sum(axis=1), "n_interacoes", label=lambda c: c[6:])
    t.to_csv(C.TAB / "ares_Q4_atividades_engajamento.csv", index=False)
    R["Q4"] = {"populacao": len(log),
               "indice_bruto": "média dos percentis de interações, sessões, dias ativos e tempo (contém efeito mecânico)",
               "indice_corrigido": "média dos percentis de sessões e dias ativos; ρ parcial controlando log(nº de interações)",
               "atividades": t.to_dict("records")}
    q4_figura(t, "ARES — Q4: tipos de interação associados ao engajamento",
              f"n = {len(log)} alunos · barra clara = não significativo após correção de Holm (10 testes)",
              FIG / "Q4_atividades_engajamento.png")


# ============================================================ Q5
def q5():
    out = {"populacao": len(perf), "evolucao": "nota_pos − nota_pre (−11 a +11); ganho normalizado de Hake",
           "correlacoes_evolucao": {}, "correlacoes_ganho_normalizado": {}, "parciais_evolucao_controle_pre": {}}
    for c in FREQ:
        out["correlacoes_evolucao"][c] = spearman(perf[c], perf["evolucao"])
        out["correlacoes_ganho_normalizado"][c] = spearman(perf[c], perf["ganho_normalizado"])
        out["parciais_evolucao_controle_pre"][c] = partial_spearman(perf[c], perf["evolucao"], perf["nota_pre"])
    out["grupos_quartil_interacoes"] = groups_test(perf, "quartil_interacoes", "evolucao")
    out["pct_melhorou_por_quartil"] = (perf.groupby("quartil_interacoes", observed=True)["evolucao"]
                                       .apply(lambda s: round(float((s > 0).mean() * 100), 1)).to_dict())
    out["regressao_a_media_rho_pre_vs_evolucao"] = spearman(perf["nota_pre"], perf["evolucao"], boot=0)["rho"]
    R["Q5"] = out

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    g = perf.groupby("quartil_interacoes", observed=True)[["nota_pre", "nota_pos"]].mean()
    for i, (a, b) in enumerate(g.values):
        ax.plot([a, b], [i, i], color=NEUTRAL, lw=2, zorder=1)
        ax.scatter(a, i, s=60, color="white", edgecolor=MUTED, linewidth=1.5, zorder=2)
        ax.scatter(b, i, s=60, color=ARES, zorder=3)
        ax.text(max(a, b) + 0.15, i, f"{b - a:+.2f}".replace(".", ","), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(4), QLAB)
    ax.set_xlabel("Nota média")
    ax.set_title("Pré (○) → pós (●) por quartil de interação")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(g.values.min() - 0.5, 3), g.values.max() + 1.2)
    ax = axs[1]
    rng = np.random.default_rng(2)
    ax.scatter(perf["n_interacoes"], perf["evolucao"] + rng.uniform(-0.2, 0.2, len(perf)), s=22, color=ARES,
               alpha=0.65, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Nº de interações de aprendizagem (escala log)")
    ax.set_ylabel("Evolução (pós − pré)")
    r = out["correlacoes_evolucao"]["n_interacoes"]
    ax.set_title("Interações × evolução")
    ax.text(0.02, 0.97, f"Spearman ρ = {r['rho']:.2f} ({r['forca']}) · {fmt_p(r['p'])}".replace(".", ",", 1),
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    fig.suptitle("ARES — Q5: frequência de interação × evolução do desempenho", x=0.0, ha="left",
                 fontweight="bold", y=1.03)
    save(fig, FIG / "Q5_interacao_evolucao.png", f"n = {len(perf)} alunos com pré e pós-teste")


# ============================================================ Q10
def q10():
    tot_ev = log[ATIV].sum().sum()
    vol = pd.DataFrame({"pre": perf["nota_pre"], "log_volume": np.log(perf["n_interacoes"])})
    rows = []
    for c in ATIV:
        r = spearman(perf[c], perf["nota_pos"], boot=0)
        rp = partial_spearman(perf[c], perf["nota_pos"], perf["nota_pre"])
        rv = partial_spearman(perf[c], perf["nota_pos"], vol)
        re = spearman(perf[c], perf["evolucao"], boot=0)
        rows.append({"atividade": c[6:], "eventos_total": int(log[c].sum()),
                     "pct_eventos": round(float(log[c].sum() / tot_ev * 100), 2),
                     "pct_alunos_que_usam": round(float((log[c] > 0).mean() * 100), 1),
                     "pct_alunos_que_usam_141": round(float((perf[c] > 0).mean() * 100), 1),
                     "media_por_aluno": round(float(log[c].mean()), 1),
                     "rho_nota_pos": r["rho"], "p_nota_pos": r["p"],
                     "rho_parcial_nota_pos_ctrl_pre": rp["rho_parcial"], "p_parcial": rp["p"],
                     "rho_parcial_ctrl_pre_e_volume": rv["rho_parcial"], "p_parcial_ctrl_volume": rv["p"],
                     "rho_evolucao": re["rho"], "p_evolucao": re["p"]})
    t = pd.DataFrame(rows)
    t["p_parcial_holm"], t["p_parcial_bh"] = ajuste_multiplo(t["p_parcial"])
    t["p_parcial_ctrl_volume_holm"], _ = ajuste_multiplo(t["p_parcial_ctrl_volume"])
    t = t.sort_values("eventos_total", ascending=False)
    t.to_csv(C.TAB / "ares_Q10_atividades_desempenho.csv", index=False)
    R["Q10"] = {"populacao_uso": len(log), "populacao_desempenho": len(perf),
                "denominadores": {"pct_eventos": f"{int(tot_ev)} eventos de aprendizagem dos {len(log)} alunos "
                                                 "(sem alertas/erros do sistema)",
                                  "pct_alunos_que_usam": f"{len(log)} alunos com log",
                                  "correlacoes": f"{len(perf)} alunos com pré, pós e log"},
                "correcao_multipla": "Holm sobre os 10 testes de cada coluna",
                "atividades": t.to_dict("records")}

    t = t.sort_values("eventos_total").reset_index(drop=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ax = axs[0]
    ax.barh(range(len(t)), t["pct_eventos"], color=ARES, height=0.65)
    for i, (v, u) in enumerate(zip(t["pct_eventos"], t["pct_alunos_que_usam"])):
        ax.text(v + 0.5, i, f"{v:.1f}% · usada por {u:.0f}% dos alunos".replace(".", ","), va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(range(len(t)), t["atividade"])
    ax.set_xlabel(f"% dos {int(tot_ev):,} eventos de aprendizagem".replace(",", "."))
    ax.set_title("Frequência de utilização")
    ax.set_xlim(0, t["pct_eventos"].max() * 1.9)
    ax.grid(axis="y", visible=False)
    q10_painel(axs[1], t, "rho_parcial_nota_pos_ctrl_pre", "p_parcial_holm", "rho_parcial_ctrl_pre_e_volume",
               "ρ parcial: uso × nota pós (controlando o pré-teste)")
    fig.suptitle("ARES — Q10: atividades mais usadas e sua relação com o desempenho", x=0.0, ha="left",
                 fontweight="bold", y=1.03)
    save(fig, FIG / "Q10_atividades_desempenho.png",
         f"uso: n = {len(log)} alunos · desempenho: n = {len(perf)} · barra clara = não significativa após correção de "
         "Holm (10 testes)")


if __name__ == "__main__":
    descritiva(); q1(); q4(); q5(); q10()
    dump(R, C.TAB / "ares_resultados.json")
    print("ARES OK")
