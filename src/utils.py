"""Funções compartilhadas: estatística (Etapa 5 — avaliação) e estilo dos gráficos."""
import json
import sys

import matplotlib

NOTEBOOK = "ipykernel" in sys.modules          # dentro do Jupyter: mostra os gráficos na célula
if not NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------- Paleta (identidade fixa por dataset) ----------------
ARES = "#2a78d6"      # azul  — slot categórico 1
EDNET = "#eb6834"     # laranja — slot categórico 2
POS = "#2a78d6"       # divergente: correlação positiva
NEG = "#e34948"       # divergente: correlação negativa
NEUTRAL = "#c3c2b7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def save(fig, path, subtitle=None):
    if subtitle:
        fig.text(0.0, -0.02, subtitle, fontsize=8.5, color=MUTED, ha="left", va="top", transform=fig.transFigure)
    if path is not None:
        fig.savefig(path)
    if NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


# ---------------- Estatística ----------------
def force(rho):
    a = abs(rho)
    if a < 0.1:
        return "desprezível"
    if a < 0.3:
        return "fraca"
    if a < 0.5:
        return "moderada"
    return "forte"


def spearman(x, y, boot=1000, seed=42):
    """Correlação de Spearman com p-valor e IC95% por bootstrap."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    rho, p = stats.spearmanr(d.x, d.y)
    lo = hi = np.nan
    if boot and n >= 10:
        rng = np.random.default_rng(seed)
        xs, ys = d.x.to_numpy(), d.y.to_numpy()
        m = min(n, 20000)  # bases grandes: bootstrap em subamostras de 20 mil
        bs = []
        for _ in range(boot if n <= 20000 else 200):
            i = rng.integers(0, n, m)
            bs.append(stats.spearmanr(xs[i], ys[i])[0])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"n": n, "rho": round(float(rho), 4), "p": float(p), "ic95": [round(float(lo), 4), round(float(hi), 4)],
            "forca": force(rho), "significativo_5pct": bool(p < 0.05)}


def partial_spearman(x, y, z):
    """Spearman parcial de x e y controlando z (correlação entre resíduos dos postos).

    z pode ser uma série (um controle) ou um DataFrame (vários controles)."""
    zs = z.to_frame() if isinstance(z, pd.Series) else pd.DataFrame(z)
    zs.columns = [f"z{i}" for i in range(zs.shape[1])]
    d = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y"), zs], axis=1).dropna()
    r = d.rank()
    Z = np.column_stack([np.ones(len(r)), r[zs.columns].to_numpy()])
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    rho = float(np.corrcoef(res(r["x"].to_numpy()), res(r["y"].to_numpy()))[0, 1])
    gl = len(d) - 2 - zs.shape[1]                      # graus de liberdade da correlação parcial: n − 2 − k
    p = float(2 * stats.t.sf(abs(rho) * np.sqrt(gl / max(1e-12, 1 - rho * rho)), gl))
    return {"n": len(d), "rho_parcial": round(rho, 4), "p": p, "forca": force(rho)}


def ajuste_multiplo(pvals):
    """p-valores ajustados por Holm (controle do erro familiar) e Benjamini-Hochberg (FDR)."""
    p = np.asarray(pvals, float)
    m, o = len(p), np.argsort(p)
    holm = np.empty(m)
    holm[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    bh = np.empty(m)
    bh[o] = np.minimum(1, np.minimum.accumulate(((m / np.arange(1, m + 1)) * p[o])[::-1])[::-1])
    return holm, bh


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def groups_test(df, group_col, y):
    """Kruskal-Wallis entre grupos + Mann-Whitney/Cliff entre o grupo mais alto e o mais baixo."""
    d = df[[group_col, y]].dropna()
    labels = sorted(d[group_col].unique())
    samples = [d.loc[d[group_col] == g, y] for g in labels]
    kw = stats.kruskal(*samples)
    hi, lo = samples[-1], samples[0]
    mw = stats.mannwhitneyu(hi, lo)
    return {
        "grupos": {str(g): {"n": int(len(s)), "media": round(float(s.mean()), 4), "mediana": round(float(s.median()), 4)}
                   for g, s in zip(labels, samples)},
        "kruskal_H": round(float(kw.statistic), 3), "kruskal_p": float(kw.pvalue),
        "mannwhitney_p_alto_vs_baixo": float(mw.pvalue),
        "cliffs_delta_alto_vs_baixo": round(float(cliffs_delta(hi, lo)), 4),
    }


def quartiles(s, labels=("1º quartil (menos)", "2º quartil", "3º quartil", "4º quartil (mais)")):
    """Quartis por posto (estável com muitos empates)."""
    return pd.qcut(s.rank(method="first"), 4, labels=list(labels))


def engagement_index(df, cols):
    """Índice de engajamento 0–100 = média dos percentis das métricas de engajamento."""
    return df[cols].rank(pct=True).mean(axis=1) * 100  # sem arredondar: arredondar cria empates artificiais


def q4_tabela(d, ativ, total, n_col, label=lambda c: c):
    """Q4 — participação de cada atividade × engajamento, em duas versões.

    bruto     : índice com volume (interações, sessões, dias, tempo). Contém associação mecânica:
                o volume é a soma das próprias atividades e a participação de uma atividade de
                contagem quase fixa (tutorial, diagnóstico) é ~k/N, caindo sozinha quando N sobe.
    corrigido : índice só de persistência (sessões + dias ativos), com correlação parcial que
                controla log(volume total). Pergunta: com o mesmo volume, quem dedica mais do seu
                uso a esta atividade volta à plataforma mais vezes?
    """
    d = d.copy()
    d["_ib"] = engagement_index(d, [n_col, "n_sessoes", "dias_ativos", "tempo_estudo_min"])
    d["_it"] = engagement_index(d, ["n_sessoes", "dias_ativos"])
    d["_g"] = quartiles(d["_it"], ("Baixo", "Médio-baixo", "Médio-alto", "Alto"))
    logn = np.log(d[n_col])
    rows = []
    for c in ativ:
        share = d[c] / total
        b = spearman(share, d["_ib"], boot=0)
        k = partial_spearman(share, d["_it"], logn)
        rows.append({"atividade": label(c), "coluna": c,
                     "rho_bruto": b["rho"], "p_bruto": b["p"],
                     "rho_participacao_x_volume": spearman(share, d[n_col], boot=0)["rho"],
                     "rho_corrigido": k["rho_parcial"], "p_corrigido": k["p"], "forca_corrigido": k["forca"],
                     "pct_uso_alto_eng": round(float((d.loc[d["_g"] == "Alto", c] > 0).mean() * 100), 1),
                     "pct_uso_baixo_eng": round(float((d.loc[d["_g"] == "Baixo", c] > 0).mean() * 100), 1)})
    t = pd.DataFrame(rows)
    t["p_corrigido_holm"], t["p_corrigido_bh"] = ajuste_multiplo(t["p_corrigido"])
    return t.sort_values("rho_corrigido").reset_index(drop=True)


def q4_figura(t, title, subtitle, path):
    """Barra = ρ corrigido (clara se não significativa após Holm); ○ = ρ bruto (com associação mecânica)."""
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.6))
    for i, r in t.iterrows():
        v = r["rho_corrigido"]
        ax.barh(i, v, color=POS if v > 0 else NEG, alpha=1 if r["p_corrigido_holm"] < 0.05 else 0.35, height=0.6)
        ax.scatter(r["rho_bruto"], i, s=42, facecolor="white", edgecolor=MUTED, linewidth=1.4, zorder=3)
        ax.plot([v, r["rho_bruto"]], [i, i], color=NEUTRAL, lw=0.8, zorder=2)
        ax.text(v + (0.02 if v >= 0 else -0.02), i - 0.28, f"{v:+.2f}".replace(".", ","), va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(t)), t["atividade"])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("ρ: participação da atividade × engajamento")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=42, facecolor="white", edgecolor=MUTED, label="bruto (índice com volume — inclui efeito mecânico)")
    ax.barh([], [], color=POS, label="corrigido (persistência, controlando o volume)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.0, 0.97), ncol=2, fontsize=8)
    fig.suptitle(title, x=0.0, ha="left", fontweight="bold", y=1.01)
    save(fig, path, subtitle)


def q10_painel(ax, t, bar, sig, mark, xlabel):
    """Barra = ρ parcial controlando o nível inicial (clara se não significativa); ○ = também controlando o volume."""
    for i, r in enumerate(t.itertuples()):
        v, m = getattr(r, bar), getattr(r, mark)
        ax.barh(i, v, color=POS if v > 0 else NEG, alpha=1 if getattr(r, sig) < 0.05 else 0.35, height=0.6)
        ax.scatter(m, i, s=38, facecolor="white", edgecolor=MUTED, linewidth=1.4, zorder=3)
        ax.text(max(v, m, 0) + 0.02 if v >= 0 else min(v, m, 0) - 0.02, i, f"{v:+.2f}".replace(".", ","),
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    lim = max(0.3, float(np.nanmax(np.abs(t[[bar, mark]].to_numpy()))) * 1.35)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(xlabel)
    ax.set_title("Relação com o desempenho")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=38, facecolor="white", edgecolor=MUTED, label="controlando também o volume total")
    ax.figure.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.0, 1.06), fontsize=8)


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def fmt_p(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")
