"""Modelo 2 (ARES) — o comportamento na plataforma ajuda a prever o desempenho no pós-teste?

Versão preditiva da Q1/Q5 do 1º bimestre. Lá, a correlação entre frequência de interação e nota foi desprezível
(ρ = 0,08). Aqui a pergunta é: um classificador que conhece o pré-teste fica MELHOR quando recebe também as
variáveis de interação e de motivação? Se não ficar, o resultado confirma, por outro caminho, a conclusão anterior.

Entrada : data/processed/ares_alunos.csv (1 linha por aluno, gerado por ares_01_prepare.py)
Saídas  : outputs/tables/ml_ares.json, outputs/figures/ml/B*.png

Amostra pequena (n = 141): validação cruzada estratificada REPETIDA (5 dobras × 10 repetições = 50 avaliações) em vez
de um único holdout, que teria só ~28 alunos no teste e variaria muito com o sorteio.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402
from ml_utils import ARES, INK2, MODEL_COLORS, NEG, POS, dump_json, save  # noqa: E402

FIG = C.FIG / "ml"
FIG.mkdir(parents=True, exist_ok=True)
RS = C.RANDOM_STATE
R = {}

# ============================================================ 1. Dados, faltantes e população
a = pd.read_csv(C.PROCESSED / "ares_alunos.csv")
MOT = ["intrinsic_goal", "extrinsic_goal", "task_value", "control_learning_beliefs", "self_efficacy", "test_anxiety"]
NOMES_MOT = {"intrinsic_goal": "Motivação intrínseca", "extrinsic_goal": "Motivação extrínseca",
             "task_value": "Valor da tarefa", "control_learning_beliefs": "Crença de controle",
             "self_efficacy": "Autoeficácia", "test_anxiety": "Ansiedade em provas"}
ATIV = [c for c in a.columns if c.startswith("ativ::")]
INTER = ["n_interacoes", "n_sessoes", "dias_ativos", "tempo_estudo_min", "n_logins", "tarefas_submetidas"] + ATIV

R["faltantes_192_alunos"] = {c: int(v) for c, v in a[["nota_pre", "nota_pos"] + MOT].isna().sum().items()}
d = a[a["tem_log"] & a["nota_pre"].notna() & a["nota_pos"].notna()].copy()
R["populacao"] = {"alunos_validos": int(len(a)), "com_pre_pos_log": int(len(d)),
                  "motivacao_faltante": int(d[MOT[0]].isna().sum()),
                  "regra": "alvo (nota no pós) faltante não é imputado: o aluno sai da modelagem. Motivação "
                           "faltante (questionário não respondido) é imputada pela mediana DENTRO de cada dobra"}
# transformação: contagens muito assimétricas → log(1 + x)
for c in INTER:
    d[c] = np.log1p(d[c])
y = (d["nota_pos"] >= 6).astype(int).to_numpy()      # 6 de 11 pontos ≈ 55% da prova
R["alvo"] = {"definicao": "aprovado = nota no pós-teste ≥ 6 (de 0 a 11)", "positivos": int(y.sum()),
             "negativos": int((1 - y).sum()), "proporcao_positivos": round(float(y.mean()), 4)}

CONJUNTOS = {
    "Pré-teste": ["nota_pre"],
    "Pré + motivação": ["nota_pre"] + MOT,
    "Interações": INTER,
    "Pré + interações": ["nota_pre"] + INTER,
    "Todos": ["nota_pre"] + MOT + INTER,
}


def modelos():
    imp = ("imp", SimpleImputer(strategy="median"))
    return {
        "Baseline (classe majoritária)": Pipeline([imp, ("clf", DummyClassifier(strategy="prior"))]),
        "Regressão Logística": Pipeline([imp, ("sc", StandardScaler()),
                                         ("clf", LogisticRegression(C=0.5, max_iter=2000))]),
        "Árvore de Decisão": Pipeline([imp, ("clf", DecisionTreeClassifier(max_depth=3, min_samples_leaf=8,
                                                                           random_state=RS))]),
        "Random Forest": Pipeline([imp, ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                                       random_state=RS))]),
    }


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS)
SCORING = {"roc_auc": "roc_auc", "acuracia_balanceada": "balanced_accuracy", "f1": "f1"}

# ============================================================ 2. Validação cruzada repetida
print("ARES — validação cruzada repetida (5×10)…", flush=True)
linhas, folds = [], {}
for cj, F in CONJUNTOS.items():
    for nome, m in modelos().items():
        if nome.startswith("Baseline") and cj != "Pré-teste":
            continue
        res = cross_validate(m, d[F], y, cv=cv, scoring=SCORING)
        folds[(cj, nome)] = res["test_roc_auc"]
        linhas.append({"conjunto": cj, "modelo": nome, "n_atributos": len(F),
                       **{f"{k}_media": round(float(np.mean(res[f"test_{k}"])), 4) for k in SCORING},
                       **{f"{k}_dp": round(float(np.std(res[f"test_{k}"])), 4) for k in SCORING}})
        print(f"  {cj:18s} {nome:30s} AUC={np.mean(res['test_roc_auc']):.3f}", flush=True)
tab = pd.DataFrame(linhas)
R["resultados_cv"] = tab.to_dict(orient="records")
melhor = tab.sort_values("roc_auc_media", ascending=False).iloc[0]
R["melhor_combinacao"] = {"conjunto": melhor.conjunto, "modelo": melhor.modelo,
                          "roc_auc": float(melhor.roc_auc_media), "dp": float(melhor.roc_auc_dp)}


# ============================================================ 3. Ganho de acrescentar comportamento (comparação pareada)
def ganho(base, novo, modelo="Regressão Logística"):
    diff = folds[(novo, modelo)] - folds[(base, modelo)]
    return {"base": base, "novo": novo, "modelo": modelo, "ganho_auc_medio": round(float(diff.mean()), 4),
            "ic95_percentil_dobras": [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])],
            "pct_dobras_com_ganho": round(float((diff > 0).mean() * 100), 1)}


R["ganho_pareado"] = [ganho("Pré-teste", "Pré + interações"), ganho("Pré-teste", "Pré + motivação"),
                      ganho("Pré-teste", "Todos"),
                      ganho("Pré-teste", "Pré + interações", "Random Forest")]

# ============================================================ 4. Sensibilidade ao tratamento dos faltantes
print("Sensibilidade da imputação…", flush=True)
F = CONJUNTOS["Pré + motivação"]
sens = {}
for rot, imp in [("mediana", SimpleImputer(strategy="median")), ("KNN (k=5)", KNNImputer(n_neighbors=5))]:
    m = Pipeline([("sc0", StandardScaler()), ("imp", imp), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    r = cross_validate(m, d[F], y, cv=cv, scoring="roc_auc")
    sens[rot] = {"n": int(len(d)), "roc_auc": round(float(r["test_score"].mean()), 4),
                 "dp": round(float(r["test_score"].std()), 4)}
cc = d[F].notna().all(axis=1).to_numpy()
m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
r = cross_validate(m, d.loc[cc, F], y[cc], cv=cv, scoring="roc_auc")
sens["exclusão dos casos incompletos"] = {"n": int(cc.sum()), "roc_auc": round(float(r["test_score"].mean()), 4),
                                          "dp": round(float(r["test_score"].std()), 4)}
R["sensibilidade_imputacao"] = sens

# ============================================================ 5. Coeficientes do modelo linear completo (interpretação)
NOMES = {"nota_pre": "Nota no pré-teste", "n_interacoes": "Nº de interações", "n_sessoes": "Nº de sessões",
         "dias_ativos": "Dias ativos", "tempo_estudo_min": "Tempo de estudo", "n_logins": "Nº de logins",
         "tarefas_submetidas": "Tarefas submetidas", **NOMES_MOT, **{c: c.replace("ativ::", "") for c in ATIV}}
Fall = CONJUNTOS["Todos"]
lr = modelos()["Regressão Logística"].fit(d[Fall], y)
coef = pd.Series(lr.named_steps["clf"].coef_[0], Fall)
R["coeficientes_lr_todos"] = {k: round(float(v), 4) for k, v in coef.sort_values(key=abs, ascending=False).items()}

# ============================================================ 6. Figuras
# B0 — dados faltantes (192 alunos) e correlações (141)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
miss = a[["nota_pre", "nota_pos"] + MOT].isna().sort_values(["nota_pos", "intrinsic_goal"])
sns.heatmap(miss.T.astype(int), cmap=sns.light_palette(ARES, as_cmap=True), cbar=False, ax=axs[0],
            xticklabels=False, yticklabels=["Nota pré", "Nota pós"] + [NOMES_MOT[c] for c in MOT])
axs[0].set_xlabel(f"{len(a)} alunos válidos (azul = valor faltante)")
axs[0].set_title("ARES — mapa de dados faltantes", fontsize=10)
axs[0].grid(False)
cols = ["nota_pre", "nota_pos", "evolucao", "n_interacoes", "n_sessoes", "dias_ativos", "tempo_estudo_min",
        "intrinsic_goal", "self_efficacy", "test_anxiety"]
lab = ["Pré", "Pós", "Evolução", "Interações", "Sessões", "Dias ativos", "Tempo", "Mot. intrínseca",
       "Autoeficácia", "Ansiedade"]
cor = d[cols].corr(method="spearman")
mask = np.triu(np.ones_like(cor, dtype=bool), k=1)
sns.heatmap(cor, mask=mask, annot=cor.map(lambda v: f"{v:.2f}".replace(".", ",")), fmt="", vmin=-1, vmax=1,
            cmap=sns.diverging_palette(10, 250, s=70, l=55, as_cmap=True), ax=axs[1], cbar_kws={"shrink": .7},
            xticklabels=lab, yticklabels=lab, annot_kws={"fontsize": 7})
axs[1].set_title(f"Correlação de Spearman (n = {len(d)})", fontsize=10)
axs[1].grid(False)
axs[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
save(fig, FIG / "B0_ares_faltantes_correlacoes.png")

# B1 — ROC-AUC por conjunto de atributos e modelo
rows = [{"Conjunto": cj, "Modelo": nm, "ROC-AUC": v} for (cj, nm), arr in folds.items() for v in arr
        if not nm.startswith("Baseline")]
fig, ax = plt.subplots(figsize=(9.2, 4))
sns.pointplot(data=pd.DataFrame(rows), x="Conjunto", y="ROC-AUC", hue="Modelo", palette=MODEL_COLORS,
              errorbar=("pi", 50), dodge=0.45, markers="o", linestyles="none", ax=ax, err_kws={"linewidth": 1.4})
ax.axhline(0.5, ls="--", color="#c3c2b7", lw=1)
ax.text(-0.45, 0.505, "acaso (AUC = 0,5)", fontsize=8, color=INK2, ha="left", va="bottom")
ax.set_ylim(0.3, 0.9)
ax.set_xlabel("")
ax.set_title("ARES — previsão de aprovação no pós-teste (validação cruzada 5×10)")
ax.legend(fontsize=8, loc="upper right", ncol=3)
save(fig, FIG / "B1_ares_auc_por_conjunto.png",
     "Ponto = média do ROC-AUC nas 50 avaliações; barra = intervalo interquartil. Aprovado = nota no pós ≥ 6.")

# B2 — coeficientes padronizados da Regressão Logística (todos os atributos)
cf = coef.sort_values()
fig, ax = plt.subplots(figsize=(7.8, 5.2))
sns.barplot(x=cf.values, y=[NOMES.get(k, k) for k in cf.index], palette=[POS if v > 0 else NEG for v in cf.values],
            hue=[NOMES.get(k, k) for k in cf.index], legend=False, ax=ax)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Coeficiente padronizado (log-odds por 1 desvio-padrão)")
ax.set_title("ARES — Regressão Logística com todos os atributos")
save(fig, FIG / "B2_ares_coeficientes.png", "Ajuste em todos os 141 alunos, regularização L2 (C = 0,5).")

dump_json(R, C.TAB / "ml_ares.json")
print("\nMelhor:", R["melhor_combinacao"])
for g in R["ganho_pareado"]:
    print(f"Ganho {g['base']} → {g['novo']} ({g['modelo']}): {g['ganho_auc_medio']:+.3f} "
          f"(IC dobras {g['ic95_percentil_dobras']}, {g['pct_dobras_com_ganho']}% das dobras)")
print("Imputação:", R["sensibilidade_imputacao"])
