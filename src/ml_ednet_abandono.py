"""Modelo 1 (EdNet) — previsão de abandono logo após o diagnóstico inicial.

CRISP-DM, 2º bimestre: Preparação → Modelagem → Avaliação → Implantação.

Pergunta de negócio: no momento em que o estudante sai do diagnóstico, é possível prever se ele vai voltar para
estudar? No 1º bimestre, o achado descritivo mais forte foi que 62,5% dos estudantes do EdNet nunca passaram do
diagnóstico. Um classificador de risco permitiria agir (lembrete, convite para a 1ª aula) antes de perder o aluno.

Entrada : data/processed/ednet_users.csv.gz (1 linha por estudante, 297.915 estudantes)
Saídas  : outputs/tables/ml_abandono.json, outputs/figures/ml/A*.png, models/modelo_abandono.joblib

Cuidados contra vazamento de informação (data leakage): só entram atributos conhecidos ao fim do diagnóstico.
Ficam de fora pct_mobile, tempo_medio_bloco_seg, n_sessoes etc., porque são medidos no histórico inteiro do aluno
(incluindo o que ele fez depois) — usá-los inflaria o desempenho do modelo com informação do futuro.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402
from ml_utils import (EDNET, INK2, MODEL_COLORS, MUTED, NEUTRAL, Cronometro, dump_json, limiar_youden,  # noqa: E402
                      metricas, resumo_cv, save)

FIG = C.FIG / "ml"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = C.ROOT / "models"
MODELS.mkdir(exist_ok=True)
RS = C.RANDOM_STATE
N_CV = 60_000          # subamostra estratificada do treino para comparar modelos (1 CPU: mantém o tempo em minutos)
R = {}

# ============================================================ 1. Dados e variável-alvo
u = pd.read_csv(C.PROCESSED / "ednet_users.csv.gz",
                usecols=["user_id", "n_eventos", "src_diagnosis", "resp_diagnosis", "acerto_diagnosis",
                         "primeiro_evento"])
u["abandono"] = (u["n_eventos"] <= u["src_diagnosis"]).astype(int)   # nunca fez nada fora do diagnóstico

# ============================================================ 2. Preparação dos atributos (Pandas/NumPy)
ts = pd.to_datetime(u["primeiro_evento"], unit="ms", utc=True)
X = pd.DataFrame({
    "questoes_diag": u["resp_diagnosis"],                         # nº de questões respondidas no diagnóstico
    "acertos_diag": u["acerto_diagnosis"],
    "acuracia_diag": u["acerto_diagnosis"] / u["resp_diagnosis"].replace(0, np.nan),     # NaN se pulou o diagnóstico
    "eventos_diag": u["src_diagnosis"],                           # cliques/ações dentro do diagnóstico
    "eventos_por_questao_diag": u["src_diagnosis"] / u["resp_diagnosis"].replace(0, np.nan),
    "pulou_diagnostico": (u["resp_diagnosis"] == 0).astype(int),  # indicador explícito do valor faltante
    "hora_inicio": ts.dt.hour,                                    # UTC
    "dia_semana": ts.dt.dayofweek,
    "mes_coorte": (ts.dt.year - 2018) * 12 + ts.dt.month - 8,     # meses desde ago/2018 (início da coleta)
})
y = u["abandono"].to_numpy()
FEATS = list(X.columns)
CONTAGENS = ["questoes_diag", "acertos_diag", "eventos_diag", "eventos_por_questao_diag"]
OUTRAS = [c for c in FEATS if c not in CONTAGENS]

faltantes = X.isna().sum()
R["dados"] = {
    "n_estudantes": int(len(u)),
    "taxa_abandono": round(float(y.mean()), 4),
    "classes": {"abandono (1)": int(y.sum()), "continuou (0)": int((1 - y).sum())},
    "razao_desbalanceamento": round(float(y.mean() / (1 - y.mean())), 2),
    "atributos": FEATS,
    "faltantes_por_atributo": {k: int(v) for k, v in faltantes.items() if v},
    "tratamento_faltantes": "acurácia e eventos/questão ficam indefinidos para quem pulou o diagnóstico "
                            "(0 questões): imputação pela mediana do treino + indicador explícito `pulou_diagnostico`",
    "excluidos_por_vazamento": ["pct_mobile", "tempo_medio_bloco_seg", "n_sessoes", "dias_ativos", "tempo_estudo_min",
                                "acessos_*", "src_* fora do diagnóstico"],
}
# taxa de abandono por nº de questões do diagnóstico (achado central, usado no artigo)
tab_q = (pd.DataFrame({"q": X["questoes_diag"].clip(upper=12), "y": y}).groupby("q")["y"]
         .agg(["mean", "size"]).rename(columns={"mean": "taxa_abandono", "size": "n"}))
R["abandono_por_questoes_diag"] = {int(k): {"taxa": round(float(r.taxa_abandono), 4), "n": int(r.n)}
                                   for k, r in tab_q.iterrows()}
grp = pd.cut(X["questoes_diag"], [-1, 0, 6, 10, 10_000], labels=["0 (pulou)", "1–6", "7–10", "11+"])
R["abandono_por_faixa"] = {str(k): {"taxa": round(float(v.mean()), 4), "n": int(len(v))}
                           for k, v in pd.Series(y).groupby(grp.to_numpy(), observed=True)}

# ============================================================ 3. Divisão treino / teste (holdout estratificado)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
X_cv, _, y_cv, _ = train_test_split(X_tr, y_tr, train_size=N_CV, stratify=y_tr, random_state=RS)
R["divisao"] = {"treino": int(len(y_tr)), "teste": int(len(y_te)), "subamostra_cv": int(len(y_cv)),
                "estrategia": "holdout 80/20 estratificado; comparação de modelos por validação cruzada estratificada "
                              "(5 dobras) numa subamostra de 60 mil estudantes do treino"}


# ============================================================ 4. Modelos candidatos
def pre_linear():
    return ColumnTransformer([
        ("contagens", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                                    StandardScaler()), CONTAGENS),
        ("outras", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), OUTRAS)])


def pre_arvore():
    return SimpleImputer(strategy="median")


CANDIDATOS = {
    "Baseline (classe majoritária)": Pipeline([("prep", pre_arvore()), ("clf", DummyClassifier(strategy="prior"))]),
    "Regressão Logística": Pipeline([("prep", pre_linear()), ("clf", LogisticRegression(max_iter=2000))]),
    "Árvore de Decisão": Pipeline([("prep", pre_arvore()),
                                   ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RS))]),
    "Random Forest": Pipeline([("prep", pre_arvore()),
                               ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1,
                                                              random_state=RS))]),
    "Gradient Boosting (HGB)": Pipeline([("prep", pre_arvore()),
                                         ("clf", HistGradientBoostingClassifier(random_state=RS))]),
}
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "acuracia": "accuracy",
           "acuracia_balanceada": "balanced_accuracy", "f1": "f1"}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

print("Comparação de modelos (validação cruzada, 5 dobras)…", flush=True)
R["comparacao_cv"] = {}
for nome, m in CANDIDATOS.items():
    with Cronometro() as t:
        res = cross_validate(m, X_cv, y_cv, cv=cv5, scoring=SCORING)
    R["comparacao_cv"][nome] = resumo_cv(res)
    print(f"  {nome:32s} AUC={R['comparacao_cv'][nome]['roc_auc']['media']:.3f}  ({t.s}s)", flush=True)

# ============================================================ 5. Desbalanceamento: peso de classe e limiar
print("Tratamento do desbalanceamento…", flush=True)
R["desbalanceamento"] = {}
for rotulo, cw in [("sem ajuste", None), ("class_weight='balanced'", "balanced")]:
    m = Pipeline([("prep", pre_arvore()), ("clf", HistGradientBoostingClassifier(class_weight=cw, random_state=RS))])
    p = cross_val_predict(m, X_cv, y_cv, cv=cv5, method="predict_proba")[:, 1]
    lim = limiar_youden(y_cv, p)
    R["desbalanceamento"][rotulo] = {"limiar_0.5": metricas(y_cv, p, 0.5, "abandono"),
                                     "limiar_otimo": metricas(y_cv, p, lim, "abandono")}

# Decisão: o class_weight='balanced' e o ajuste do limiar chegam à MESMA acurácia balanceada (ver JSON), mas o peso
# de classe distorce as probabilidades (Brier pior). Como a API devolve a probabilidade de abandono, o modelo final
# fica sem peso de classe e o desbalanceamento é tratado pelo limiar de decisão.

# ============================================================ 6. Ajuste de hiperparâmetros (Grid Search)
print("Grid search do HGB…", flush=True)
grid = GridSearchCV(
    Pipeline([("prep", pre_arvore()), ("clf", HistGradientBoostingClassifier(random_state=RS))]),
    {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31], "clf__min_samples_leaf": [20, 200],
     },
    scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=RS))
with Cronometro() as t:
    grid.fit(X_cv, y_cv)
R["grid_search"] = {"melhores_parametros": {k.replace("clf__", ""): v for k, v in grid.best_params_.items()},
                    "melhor_auc_cv": round(float(grid.best_score_), 4), "combinacoes": len(grid.cv_results_["params"]),
                    "tempo_s": t.s}

# limiar de decisão: escolhido em previsões fora da dobra (sem olhar o teste)
best = grid.best_estimator_
p_oof = cross_val_predict(best, X_cv, y_cv, cv=cv5, method="predict_proba")[:, 1]
LIMIAR = limiar_youden(y_cv, p_oof)
R["limiar_decisao"] = {"valor": round(LIMIAR, 3), "criterio": "máxima acurácia balanceada (Youden) em previsões "
                                                             "fora da dobra do treino"}

# ============================================================ 7. Modelo final: treino completo → teste
print("Treino final e avaliação no teste…", flush=True)
with Cronometro() as t:
    best.fit(X_tr, y_tr)
p_te = best.predict_proba(X_te)[:, 1]
R["teste"] = {"limiar_0.5": metricas(y_te, p_te, 0.5, "abandono"),
              "limiar_escolhido": metricas(y_te, p_te, LIMIAR, "abandono"), "tempo_treino_s": t.s}
# Regressão Logística no mesmo teste, para a discussão (modelo linear e interpretável)
lr = CANDIDATOS["Regressão Logística"].fit(X_tr, y_tr)
p_lr = lr.predict_proba(X_te)[:, 1]
R["teste_regressao_logistica"] = metricas(y_te, p_lr, limiar_youden(y_tr[:N_CV], lr.predict_proba(X_tr[:N_CV])[:, 1]),
                                          "abandono")

# regra simples equivalente (sanidade): "respondeu ≤ 6 questões no diagnóstico"
regra = ((X_te["questoes_diag"] >= 1) & (X_te["questoes_diag"] <= 6)).astype(int).to_numpy()
R["regra_simples_1a6_questoes"] = metricas(y_te, regra.astype(float), 0.5, "abandono")

# segmento: quem passou da 6ª questão (o modelo ainda discrimina além do ponto de ruptura?)
seg = X_te["questoes_diag"] >= 7
R["auc_segmento_7mais_questoes"] = {"n": int(seg.sum()), "taxa_abandono": round(float(y_te[seg].mean()), 4),
                                     "roc_auc": round(float(roc_auc_score(y_te[seg], p_te[seg])), 4)}

# ============================================================ 8. Validação temporal (coortes futuras)
corte = pd.Timestamp("2019-07-01", tz="UTC")
tr_t, te_t = (ts < corte).to_numpy(), (ts >= corte).to_numpy()
mt = Pipeline([("prep", pre_arvore()), ("clf", HistGradientBoostingClassifier(random_state=RS, **{
    k.replace("clf__", ""): v for k, v in grid.best_params_.items()}))]).fit(X[tr_t], y[tr_t])
p_t = mt.predict_proba(X[te_t])[:, 1]
R["validacao_temporal"] = {"treino": "coortes até jun/2019", "teste": "coortes de jul/2019 a nov/2019",
                           "n_treino": int(tr_t.sum()), "n_teste": int(te_t.sum()),
                           "metricas": metricas(y[te_t], p_t, LIMIAR, "abandono")}

# ============================================================ 9. Interpretação: importância por permutação
print("Importância por permutação…", flush=True)
rng = np.random.default_rng(RS)
idx = rng.choice(len(y_te), 30_000, replace=False)
pi = permutation_importance(best, X_te.iloc[idx], y_te[idx], scoring="roc_auc", n_repeats=5, random_state=RS)
imp = pd.Series(pi.importances_mean, FEATS).sort_values(ascending=False)
imp_sd = pd.Series(pi.importances_std, FEATS)[imp.index]
R["importancia_permutacao_auc"] = {k: {"media": round(float(v), 4), "dp": round(float(imp_sd[k]), 4)}
                                   for k, v in imp.items()}

# ============================================================ 10. Figuras (Matplotlib + Seaborn)
NOMES = {"questoes_diag": "Questões respondidas no diagnóstico", "acertos_diag": "Acertos no diagnóstico",
         "acuracia_diag": "Acurácia no diagnóstico", "eventos_diag": "Eventos no diagnóstico",
         "eventos_por_questao_diag": "Eventos por questão", "pulou_diagnostico": "Pulou o diagnóstico",
         "hora_inicio": "Hora de início", "dia_semana": "Dia da semana", "mes_coorte": "Mês de entrada (coorte)"}

# A1 — taxa de abandono por nº de questões do diagnóstico + previsão média do modelo
d = pd.DataFrame({"q": X_te["questoes_diag"].clip(upper=12), "y": y_te, "p": p_te})
g = d.groupby("q").agg(obs=("y", "mean"), prev=("p", "mean"), n=("y", "size")).reset_index()
fig, ax = plt.subplots(figsize=(8.4, 3.9))
sns.barplot(data=g, x="q", y="obs", color=EDNET, alpha=0.85, ax=ax, label="Observado (teste)")
ax.plot(range(len(g)), g["prev"], color=INK2, marker="o", lw=1.6, ms=4, label="Previsto pelo modelo")
for i, r in g.iterrows():
    ax.text(i, max(r.obs, r.prev) + 0.025, f"{r.obs * 100:.0f}%", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(range(len(g)), [str(int(q)) if q < 12 else "12+" for q in g["q"]])
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
ax.set_xlabel("Nº de questões respondidas no diagnóstico inicial")
ax.set_ylabel("Taxa de abandono")
ax.set_title("EdNet — abandono após o diagnóstico, por nº de questões respondidas")
ax.legend(loc="upper right", fontsize=8)
save(fig, FIG / "A1_abandono_por_questoes_diag.png",
     f"Conjunto de teste (n = {len(y_te):,}). Abandono = nunca usou a plataforma fora do diagnóstico.".replace(",", "."))

# A2 — comparação de modelos (validação cruzada)
rows = []
for nome, r in R["comparacao_cv"].items():
    for met, lab in [("roc_auc", "ROC-AUC"), ("acuracia_balanceada", "Acurácia balanceada"), ("f1", "F1 (abandono)")]:
        rows.append({"Modelo": nome.replace(" (classe majoritária)", ""), "Métrica": lab,
                     "Valor": r[met]["media"], "dp": r[met]["dp"]})
cmp = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8.4, 3.9))
pal = {k.replace(" (classe majoritária)", ""): v for k, v in MODEL_COLORS.items()}
sns.barplot(data=cmp, x="Métrica", y="Valor", hue="Modelo", palette=pal, ax=ax)
for cont in ax.containers:
    ax.bar_label(cont, labels=[f"{v:.2f}".replace(".", ",") for v in cont.datavalues], fontsize=6.5,
                 color=INK2, padding=1)
ax.set_ylim(0, 1.05)
ax.set_xlabel("")
ax.set_ylabel("Média em 5 dobras")
ax.set_title("EdNet — comparação dos modelos de abandono (validação cruzada)")
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, FIG / "A2_comparacao_modelos.png")

# A3 — avaliação no teste: matriz de confusão, ROC e Precisão-Recall
mc = R["teste"]["limiar_escolhido"]["matriz_confusao"]
cm = np.array([[mc["VN"], mc["FP"]], [mc["FN"], mc["VP"]]])
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=np.array(
    [[f"{v / s * 100:.1f}%".replace(".", ",") + f"\n({v:,})".replace(",", ".") for v in row] for row, s in zip(cm, cm.sum(axis=1))]),
    fmt="", cmap=sns.light_palette(EDNET, as_cmap=True), cbar=False, ax=axs[0], vmin=0, vmax=1,
    xticklabels=["Continuou", "Abandono"], yticklabels=["Continuou", "Abandono"], annot_kws={"fontsize": 8.5})
axs[0].set_xlabel("Previsto")
axs[0].set_ylabel("Real")
axs[0].set_title(f"Matriz de confusão (limiar {LIMIAR:.2f})".replace(".", ","), fontsize=10)
axs[0].grid(False)
for nome, pp, cor in [("HGB", p_te, EDNET), ("Reg. Logística", p_lr, "#2a78d6")]:
    fpr, tpr, _ = roc_curve(y_te, pp)
    axs[1].plot(fpr, tpr, color=cor, lw=1.8, label=f"{nome} (AUC {roc_auc_score(y_te, pp):.3f})".replace(".", ","))
    pr, rc, _ = precision_recall_curve(y_te, pp)
    axs[2].plot(rc, pr, color=cor, lw=1.8, label=nome)
axs[1].plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1)
axs[1].set_xlabel("Taxa de falsos positivos")
axs[1].set_ylabel("Taxa de verdadeiros positivos")
axs[1].set_title("Curva ROC", fontsize=10)
axs[1].legend(fontsize=7.5, loc="lower right")
axs[2].axhline(y_te.mean(), ls="--", color=NEUTRAL, lw=1, label=f"Prevalência ({y_te.mean() * 100:.1f}%)".replace(".", ","))
axs[2].set_xlabel("Recall (abandono)")
axs[2].set_ylabel("Precisão")
axs[2].set_title("Curva Precisão-Recall", fontsize=10)
axs[2].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
save(fig, FIG / "A3_avaliacao_teste.png")

# A4 — importância por permutação
ti = pd.DataFrame({"Atributo": [NOMES[k] for k in imp.index], "Queda no AUC": imp.values, "dp": imp_sd.values})
fig, ax = plt.subplots(figsize=(7.6, 3.6))
sns.barplot(data=ti, y="Atributo", x="Queda no AUC", color=EDNET, ax=ax)
ax.errorbar(ti["Queda no AUC"], range(len(ti)), xerr=ti["dp"], fmt="none", ecolor=INK2, lw=1)
for i, v in enumerate(ti["Queda no AUC"]):
    ax.text(v + 0.004, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=7.5, color=INK2)
ax.set_ylabel("")
ax.set_title("EdNet — importância dos atributos (permutação)")
save(fig, FIG / "A4_importancia_atributos.png",
     "Queda média do ROC-AUC ao embaralhar cada atributo (5 repetições, 30 mil estudantes do teste).")

# ============================================================ 11. Implantação: salvar o modelo
joblib.dump({"modelo": best, "atributos": FEATS, "limiar": LIMIAR, "versao": "1.0",
             "descricao": "HGB — probabilidade de abandono após o diagnóstico (EdNet-KT3)",
             "auc_teste": R["teste"]["limiar_escolhido"]["roc_auc"]}, MODELS / "modelo_abandono.joblib")
R["modelo_salvo"] = "models/modelo_abandono.joblib"
dump_json(R, C.TAB / "ml_abandono.json")

te = R["teste"]["limiar_escolhido"]
print(f"\nTeste: AUC={te['roc_auc']:.3f}  PR-AUC={te['pr_auc']:.3f}  acurácia={te['acuracia']:.3f}  "
      f"acurácia balanceada={te['acuracia_balanceada']:.3f}  F1={te['f1']:.3f}  (limiar {LIMIAR:.2f})")
print(f"Validação temporal: AUC={R['validacao_temporal']['metricas']['roc_auc']:.3f}")
print("Melhores parâmetros:", R["grid_search"]["melhores_parametros"])
