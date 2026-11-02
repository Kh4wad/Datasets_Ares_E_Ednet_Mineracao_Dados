# Datasets-Beready — Mineração de dados educacionais (CRISP-DM)

Análise dos datasets **ARES** (OSF 8y3zp) e **EdNet-KT3** (Riiid) para responder:

- **Q1** — Frequência de interação × desempenho
- **Q4** — Tipos de interação associados a maior engajamento
- **Q5** — Frequência de interação × evolução do desempenho
- **Q10** — Atividades mais usadas e sua relação com o desempenho

📄 **Documentação completa (passo a passo, validação, resultados, gráficos e comparação):** [docs/DOCUMENTACAO.md](docs/DOCUMENTACAO.md)

🎤 **Síntese para a apresentação (números confirmados, roteiro de slides, perguntas da banca):** [docs/SINTESE_APRESENTACAO.md](docs/SINTESE_APRESENTACAO.md)

## Resultado em uma linha por pergunta

*Versão revisada em 11/10/2026 (2ª auditoria): ver a seção 13 da documentação. Todos os resultados são **associações** observacionais, não relações de causa. No EdNet, quase tudo é estatisticamente significativo por causa do tamanho da base; a relevância é julgada pela magnitude do ρ.*

| | ARES | EdNet | Consolidado |
|---|---|---|---|
| Q1 | Spearman ρ = +0,08, Pearson r = +0,12 (ambos ns) | ρ = +0,15 (fraca; +0,14 controlando o diagnóstico) | Associação fraca ou não detectada |
| Q4 | Leitura de textos +0,31, navegação +0,24 (corrigido) | Aulas +0,24, questões salvas +0,24 (corrigido) | Mais uso de conteúdo de estudo ↔ volta mais vezes |
| Q5 | ρ = 0,00 | ρ = +0,04 (evolução média +1,0 p.p.) | Sem associação relevante entre interagir mais e evoluir mais |
| Q10 | Mais usada: responder questões. Nenhuma significativa após a correção de Holm (tutorial no limite, p = 0,054) | Mais usada: prática. Prática +0,16 (+0,10 controlando o volume) | Relação atividade × desempenho fraca ou inexistente |

## 2º bimestre — Modelagem com aprendizado de máquina (Scikit-Learn)

| Script | Modelo | Resultado principal |
|---|---|---|
| [src/ml_ednet_abandono.py](src/ml_ednet_abandono.py) | Risco de abandono logo após o diagnóstico (EdNet, 297.915 estudantes). Compara Baseline, Regressão Logística, Árvore, Random Forest e HistGradientBoosting; grid search; limiar ajustado ao desbalanceamento; validação temporal | HGB: ROC-AUC 0,891 no teste (0,862 em coortes futuras) |
| [src/ml_ares_desempenho.py](src/ml_ares_desempenho.py) | Aprovação no pós-teste (ARES, 141 alunos), validação cruzada 5×10, imputação dos faltantes de motivação | Pré-teste sozinho: AUC 0,67; interações não acrescentam poder preditivo |
| [api/main.py](api/main.py) | API FastAPI que consulta o modelo de abandono | `uvicorn api.main:app --reload` → http://127.0.0.1:8000/docs |

Saídas: `outputs/figures/ml/`, `outputs/tables/ml_abandono.json`, `outputs/tables/ml_ares.json`, `models/modelo_abandono.joblib`.
Testes da API: `python tests/test_api.py`. O artigo final está em `docs/artigo/`.

Exemplo de chamada:
```bash
curl -X POST http://127.0.0.1:8000/prever/abandono -H "Content-Type: application/json" \
     -d '{"questoes_respondidas": 6, "acertos": 3, "eventos": 18}'
```

## Passo a passo em notebooks

A pasta [notebooks/](notebooks/) tem o passo a passo completo da documentação, **executado**, com código, tabelas e gráficos. Abra na ordem:

| Notebook | O que mostra |
|---|---|
| [01_entendimento_dos_dados](notebooks/01_entendimento_dos_dados.ipynb) | Perguntas, arquivos, linhas, colunas, nulos, duplicadas; onde está o EdNet |
| [02_preparacao_ARES](notebooks/02_preparacao_ARES.ipynb) | Limpeza do ARES passo a passo; descoberta da regra “nota sem a Q9” |
| [03_preparacao_EdNet](notebooks/03_preparacao_EdNet.ipynb) | Um estudante tratado linha a linha; qualidade da base inteira; bases de cada pergunta |
| [04_Q1_interacao_desempenho](notebooks/04_Q1_interacao_desempenho.ipynb) | Spearman, correlação parcial à mão, Pearson 0,118 × Spearman 0,08, diagnóstico no EdNet |
| [05_Q4_engajamento](notebooks/05_Q4_engajamento.ipynb) | A associação mecânica demonstrada; índice corrigido; Holm passo a passo |
| [06_Q5_evolucao](notebooks/06_Q5_evolucao.ipynb) | Fórmulas, fluxo 297.915 → 42.720, erro da fórmula antiga demonstrado |
| [07_Q10_atividades](notebooks/07_Q10_atividades.ipynb) | Denominadores, origens × acessos, controle de volume, Holm |
| [08_comparacao_e_recomendacoes](notebooks/08_comparacao_e_recomendacoes.ipynb) | ARES × EdNet, regra de convergência, recomendações, limitações |
| [09_auditoria](notebooks/09_auditoria.ipynb) | Recálculo independente (93/93) e documento × resultados (21/21) |

Cada notebook termina **conferindo** os seus números com os resultados oficiais (`outputs/tables/`).

## Executar

```bash
pip install -r requirements.txt
python run_all.py                        # pipeline completo
python run_all.py --skip-ednet-extract   # reaproveita a extração do EdNet (~1–2 min)
python run_all.py --notebooks            # também executa os notebooks 01–09 (~6 min a mais)
```

Testado com Python 3.12 e 3.14 no Windows.

- **Sem o EdNet bruto** (fora do repositório), o `run_all.py` avisa, pula a extração e usa `data/processed/ednet_users.csv.gz`. As análises, os gráficos, a auditoria e os notebooks rodam normalmente; só as demonstrações que leem os arquivos originais são puladas, com aviso.
- **Onde colocar o EdNet** (para regenerar a extração e ver as demonstrações com os arquivos originais): `data/raw/ednet/` ou uma pasta `Datasets/` dentro ou ao lado do projeto. Aceita **`EdNet-KT3.zip` ou a pasta `EdNet-KT3/` descompactada**, e `EdNet-Contents/` (ou `contents/`). Exemplo para o layout do `Proj_mineracoa`: copie `Datasets/EdNet-KT3` e `Datasets/EdNet-Contents` para `data/raw/ednet/`, ou rode antes `$env:EDNET_DIR = "C:\caminho\Proj_mineracoa\Datasets"` (PowerShell). O notebook 01 mostra onde o projeto procurou.
- No final, `auditoria.py` recalcula de forma independente os principais resultados e confere se os números-chave aparecem na documentação. Se algo divergir, a execução termina com erro.
- Os sinais `~~~^^^` num traceback do Python só apontam o trecho da linha que falhou. A mensagem de erro é sempre a última linha.
