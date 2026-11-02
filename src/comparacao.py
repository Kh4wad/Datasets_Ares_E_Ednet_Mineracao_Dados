"""Etapa 5 do CRISP-DM — Avaliação: compara as respostas de ARES e EdNet para Q1, Q4, Q5 e Q10.

Entrada: outputs/tables/{ares,ednet}_resultados.json
Saída  : outputs/tables/comparacao.json, comparacao.md + outputs/figures/comparacao/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import ARES, EDNET, INK2, MUTED, dump, save

A = json.loads((C.TAB / "ares_resultados.json").read_text(encoding="utf-8"))
E = json.loads((C.TAB / "ednet_resultados.json").read_text(encoding="utf-8"))
FIG = C.FIG / "comparacao"
LEGENDA = [Patch(color=ARES, label="ARES"), Patch(color=EDNET, label="EdNet")]  # sempre na cor cheia

METRICS = [("Nº de interações", "n_interacoes", "n_eventos"), ("Nº de sessões", "n_sessoes", "n_sessoes"),
           ("Dias ativos", "dias_ativos", "dias_ativos")]

# Atividades equivalentes entre as plataformas (mesma função pedagógica)
EQUIV = [
    ("Responder questões", "Responder questões/tarefas", "Prática por parte (sprint)"),
    ("Ver feedback/explicação", "Visualização de feedback/nota", "Leitura de explicações (feedback)"),
    ("Estudar conteúdo", "Leitura de textos", "Aulas (vídeo/texto)"),
    ("Recursos de apoio/revisão", "Vocabulário (palavras e traduções)", "Caderno de erros/anotações"),
    ("Atividade de entrada", "Tutorial", "Diagnóstico inicial"),
]


def concl(ra, re_, pa, pe, lim=0.1):
    """Efeito = |ρ| ≥ 0,1 E p < 0,05 (p já corrigido por Holm quando há várias atividades)."""
    sinal = lambda r, p: "0" if abs(r) < lim or p >= 0.05 else "+" if r > 0 else "−"
    sa, se = sinal(ra, pa), sinal(re_, pe)
    if sa == se == "0":
        return "Convergente (sem efeito nas duas)"
    return "Convergente" if sa == se else "Parcialmente convergente" if "0" in (sa, se) else "Divergente"


def bar_pair(ax, labels, va, vb, ca=None, cb=None, title="", xlabel="ρ de Spearman", pa=None, pb=None):
    """Barras ARES × EdNet; barra clara = p ≥ 0,05 (pa/pb: p-valores, corrigidos quando há vários testes)."""
    y = np.arange(len(labels))
    h = 0.36
    alpha = lambda ps, n: [1 if ps is None or p < 0.05 else 0.35 for p in (ps or [0] * n)]
    for yy, vals, cor, nome, ps in ((y + h / 2, va, ARES, "ARES", pa), (y - h / 2, vb, EDNET, "EdNet", pb)):
        for yv, v, a in zip(yy, vals, alpha(ps, len(vals))):
            ax.barh(yv, v, height=h - 0.04, color=cor, alpha=a)
    for yy, vals, cis in ((y + h / 2, va, ca), (y - h / 2, vb, cb)):
        if cis is not None:
            lo = [v - c[0] for v, c in zip(vals, cis)]
            hi = [c[1] - v for v, c in zip(vals, cis)]
            ax.errorbar(vals, yy, xerr=[lo, hi], fmt="none", ecolor=INK2, capsize=2, lw=0.8)
        for k, (v, yv) in enumerate(zip(vals, yy)):
            edge = (cis[k][1] if v >= 0 else cis[k][0]) if cis is not None else v  # rótulo depois da barra de erro
            ax.text(edge + (0.02 if v >= 0 else -0.02), yv, f"{v:+.2f}".replace(".", ","), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=MUTED, lw=0.5, alpha=0.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(-1.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()


def main():
    out = {}
    # ---------- Q1 e Q5
    for q, ka, ke, tit in (("Q1", "correlacoes", "correlacoes", "Q1 — Frequência × desempenho"),
                           ("Q5", "correlacoes_evolucao", "correlacoes_evolucao", "Q5 — Frequência × evolução")):
        rows = []
        for lab, ma, me in METRICS:
            ra, re_ = A[q][ka][ma], E[q][ke][me]
            rows.append({"metrica": lab, "ares_rho": ra["rho"], "ares_p": ra["p"], "ares_ic95": ra["ic95"],
                         "ednet_rho": re_["rho"], "ednet_p": re_["p"], "ednet_ic95": re_["ic95"],
                         "conclusao": concl(ra["rho"], re_["rho"], ra["p"], re_["p"])})
        out[q] = rows
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, q, tit in ((axs[0], "Q1", "Q1 — Frequência × desempenho"), (axs[1], "Q5", "Q5 — Frequência × evolução")):
        r = out[q]
        bar_pair(ax, [x["metrica"] for x in r], [x["ares_rho"] for x in r], [x["ednet_rho"] for x in r],
                 [x["ares_ic95"] for x in r], [x["ednet_ic95"] for x in r], tit,
                 pa=[x["ares_p"] for x in r], pb=[x["ednet_p"] for x in r])
    fig.legend(handles=LEGENDA, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.06))
    fig.suptitle("Comparação ARES × EdNet — frequência de interação", x=0.0, ha="left", fontweight="bold", y=1.04)
    save(fig, FIG / "Q1_Q5_comparacao.png",
         "barras de erro = IC95% por bootstrap · linhas finas em ±0,1 = limite de correlação desprezível · barra clara = p ≥ 0,05")

    # ---------- Q4 e Q10 (atividades equivalentes)
    a4 = {r["atividade"]: r for r in A["Q4"]["atividades"]}
    e4 = {r["atividade"]: r for r in E["Q4"]["atividades"]}
    a10 = {r["atividade"]: r for r in A["Q10"]["atividades"]}
    e10 = {r["atividade"]: r for r in E["Q10"]["atividades"]}
    out["Q4"], out["Q10"] = [], []
    for lab, aa, ee in EQUIV:
        ra, re_ = a4[aa]["rho_corrigido"], e4[ee]["rho_corrigido"]
        out["Q4"].append({"funcao": lab, "ares_atividade": aa, "ednet_atividade": ee, "ares_rho": ra,
                          "ares_p": a4[aa]["p_corrigido"], "ednet_rho": re_, "ednet_p": e4[ee]["p_corrigido"],
                          "ares_rho_bruto": a4[aa]["rho_bruto"], "ednet_rho_bruto": e4[ee]["rho_bruto"],
                          "ares_p_holm": a4[aa]["p_corrigido_holm"], "ednet_p_holm": e4[ee]["p_corrigido_holm"],
                          "conclusao": concl(ra, re_, a4[aa]["p_corrigido_holm"], e4[ee]["p_corrigido_holm"])})
        ra, re_ = a10[aa]["rho_parcial_nota_pos_ctrl_pre"], e10[ee]["rho_parcial_acuracia_ctrl_diag"]
        out["Q10"].append({"funcao": lab, "ares_atividade": aa, "ednet_atividade": ee,
                           "ares_pct_eventos": a10[aa]["pct_eventos"], "ares_pct_usuarios": a10[aa]["pct_alunos_que_usam"],
                           "ednet_pct_eventos": e10[ee]["pct_eventos"], "ednet_pct_usuarios": e10[ee]["pct_estudantes_que_usam"],
                           "ares_rho_parcial": ra, "ares_p": a10[aa]["p_parcial"],
                           "ednet_rho_parcial": re_, "ednet_p": e10[ee]["p_parcial"], "ares_p_holm": a10[aa]["p_parcial_holm"],
                           "conclusao": concl(ra, re_, a10[aa]["p_parcial_holm"], e10[ee]["p_parcial"])})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    labs = [l for l, _, _ in EQUIV]
    bar_pair(axs[0], labs, [r["ares_rho"] for r in out["Q4"]], [r["ednet_rho"] for r in out["Q4"]],
             title="Q4 — Associação com o engajamento", xlabel="ρ corrigido: participação × persistência (controle de volume)",
             pa=[r["ares_p_holm"] for r in out["Q4"]], pb=[r["ednet_p_holm"] for r in out["Q4"]])
    bar_pair(axs[1], labs, [r["ares_rho_parcial"] for r in out["Q10"]], [r["ednet_rho_parcial"] for r in out["Q10"]],
             title="Q10 — Relação com o desempenho", xlabel="ρ parcial: uso × desempenho (controle inicial)",
             pa=[r["ares_p_holm"] for r in out["Q10"]], pb=[r["ednet_p"] for r in out["Q10"]])
    fig.legend(handles=LEGENDA, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.06))
    fig.suptitle("Comparação ARES × EdNet — atividades equivalentes", x=0.0, ha="left", fontweight="bold", y=1.04)
    save(fig, FIG / "Q4_Q10_comparacao.png",
         "barra clara = não significativa (p ≥ 0,05; ARES com correção de Holm)\nequivalências: " +
         " · ".join(f"{l}: {a} ↔ {e}" for l, a, e in EQUIV[:2]) + "\n" +
         " · ".join(f"{l}: {a} ↔ {e}" for l, a, e in EQUIV[2:]))
    dump(out, C.TAB / "comparacao.json")

    # tabela markdown para a documentação
    md = ["| Pergunta | Medida | ARES (ρ) | EdNet (ρ) | Conclusão |", "|---|---|---|---|---|"]
    f = lambda v: f"{v:+.2f}".replace(".", ",")
    for q in ("Q1", "Q5"):
        for r in out[q]:
            md.append(f"| {q} | {r['metrica']} | {f(r['ares_rho'])} | {f(r['ednet_rho'])} | {r['conclusao']} |")
    for q, ka, ke in (("Q4", "ares_rho", "ednet_rho"), ("Q10", "ares_rho_parcial", "ednet_rho_parcial")):
        for r in out[q]:
            md.append(f"| {q} | {r['funcao']} | {f(r[ka])} | {f(r[ke])} | {r['conclusao']} |")
    (C.TAB / "comparacao.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
