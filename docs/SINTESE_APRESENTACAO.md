# Síntese para a apresentação — Mineração de Dados Beready

Material de apoio para montar os slides e responder à banca. Todos os números foram **confirmados por execução** em 11/10/2026: pipeline completo, recálculo independente da auditoria com 93/93 verificações, Python 3.12 e 3.14. Detalhes em [DOCUMENTACAO.md](DOCUMENTACAO.md); o histórico de correções está na §13.

---

## 1. O que foi verificado e corrigido

| # | Problema | Efeito na conclusão |
|---|---|---|
| 1 | **Q4:** o índice de engajamento continha o volume total, a mesma variável usada para medir as atividades, o que gerava correlações artificiais | **Grande.** “Ver feedback marca os engajados” (+0,31 / +0,72) não se sustenta. O que fica é **estudar conteúdo** (+0,31 / +0,24) |
| 2 | **Q5 (EdNet):** a janela inicial da evolução tinha 36% de questões do diagnóstico, que é mais difícil | Evolução média de +5,9 → **+1,0 p.p.** A associação continua desprezível |
| 3 | **Q10:** sem correção para 10 testes, sem controle do volume e com denominadores misturados | Vocabulário deixa de ser significativo. Explicações perdem a associação ao controlar o volume. Prática: 33% → **89% dos ativos** |
| 4 | **Q1 (ARES):** Pearson 0,118 (notebook antigo) × Spearman 0,08 | Mesmos 141 alunos. A diferença é **de método**. Os dois são não significativos |
| 5 | Comparação ignorava a significância; havia linguagem causal | Critério refeito; recomendações reescritas como hipóteses |
| 6 | **Q1 (EdNet):** a acurácia misturava diagnóstico e prática, e quem usa pouco tem quase só diagnóstico | ρ 0,36 → **0,15** (de moderada para fraca) |
| 7 | O projeto não rodava em outro computador (caminho fixo; acentos quebrados) | Corrigido e testado numa cópia com caminho acentuado, com Python 3.14 |
| 8 | Arredondamento do índice e graus de liberdade da correlação parcial | Mudanças na 3ª casa decimal. Nenhuma conclusão muda |

## 2. O que foi confirmado por execução

- `python run_all.py --skip-ednet-extract` com Python 3.12: todas as etapas concluídas; recálculo independente 93/93; 21/21 números-chave encontrados na documentação.
- **Cenário A** (como na máquina do colega: Python 3.14, caminho com acento, sem o EdNet bruto): a extração é pulada com aviso e o resto conclui com 93/93.
- **Cenário B** (do zero, sem nenhuma saída gerada, Python 3.14, com o EdNet bruto): tudo foi regenerado (extração em 483 s, 13 gráficos, 93/93). Dados processados e resultados ficaram **idênticos** aos da execução com Python 3.12.
- Os 297.915 arquivos do EdNet foram reprocessados após a correção da Q5. As 46 colunas antigas ficaram idênticas; só foram acrescentadas 2.

## 3. Números que mudaram e por quê

| Medida | Antes | Agora | Por quê |
|---|---|---|---|
| EdNet Q1, ρ interações × desempenho | 0,36 | **0,15** | O desempenho passou a excluir o diagnóstico (problema 6) |
| EdNet Q5, evolução média | +5,9 p.p. | **+1,0 p.p.** | As janelas passaram a excluir o diagnóstico (problema 2) |
| EdNet Q5, n | 46.635 | **42.720** | Agora exige ≥ 40 questões *fora* do diagnóstico |
| ARES Q4, feedback | +0,31 | +0,14 (ns) | Efeito mecânico removido (problema 1) |
| EdNet Q4, explicações | +0,72 | +0,06 | Idem |
| ARES Q4, tutorial | −0,56 | −0,03 (ns) | Idem |
| ARES Q10, vocabulário | p = 0,042 | p Holm = 0,38 | Correção para 10 testes (problema 3) |
| EdNet Q10, prática usada por | 33% | 89% dos ativos | Denominador: estudantes que foram além do diagnóstico |

## 4. Limitações que permanecem

- **Tudo é associação**, sem efeito causal demonstrado. Para afirmar causa seria preciso um experimento (teste A/B).
- **ARES pequeno (n = 141):** só detecta |ρ| ≥ 0,24. “Não detectado” ≠ “não existe”.
- **EdNet sem pré/pós-teste:** o diagnóstico serve de *proxy*, e as questões são adaptativas (a dificuldade muda por aluno).
- **Q5 EdNet com viés de seleção:** só entram estudantes com ≥ 40 questões de prática (os mais engajados).
- **Populações diferentes:** leitura em inglês (ARES) × TOEIC na Coreia (EdNet). As escalas de evolução não são comparáveis.
- **Categorias de atividade do ARES** foram criadas por regras sobre os nomes de botões e diálogos.

## 5. Conclusões finais

Leitura em quatro níveis: **observado → significância → magnitude → conclusão**.

| Pergunta | ARES | EdNet | Conclusão final |
|---|---|---|---|
| **Q1** Frequência × desempenho | ρ = 0,08 · p = 0,35 · desprezível (IC até 0,24) | ρ = 0,15 · p < 0,001 · **fraca** | Associação **fraca** (EdNet) ou **não detectada** (ARES). A quantidade de interação, sozinha, prevê pouco |
| **Q4** Interações × engajamento | Leitura de textos +0,31 e navegação +0,24 (significativas após Holm). O resto ns | Aulas +0,24, questões salvas +0,24, caderno +0,17 (p < 0,001; todas **fracas**) | Quem dedica mais do uso a **conteúdo de estudo** volta mais vezes, nas duas bases. É associação, não causa |
| **Q5** Frequência × evolução | ρ = 0,00 · p = 0,99 | ρ = 0,04 · p < 0,001 · **desprezível** | **Sem associação de relevância prática** nas duas bases |
| **Q10** Atividades × desempenho | Mais usada: responder questões (48%). Nenhuma significativa após Holm | Mais usada: prática (72% dos eventos). Prática +0,16 (+0,10 controlando o volume) | Relação **fraca ou inexistente**; não há “atividade mágica” |

**Frase-resumo para o último slide:** *“Nos dois conjuntos de dados, interagir mais não está associado a aprender mais. O que acompanha o retorno do aluno é dedicar tempo a conteúdo de estudo, e o maior problema observado é o abandono logo após a primeira atividade (62,5% no EdNet). Essas são hipóteses para testar no Beready, não efeitos comprovados.”*

## 6. Sugestão de slides (números e gráficos)

| Slide | Conteúdo | Gráfico |
|---|---|---|
| 1 | Objetivo e as 4 perguntas | — |
| 2 | Dados: ARES 116.168 eventos → 104.122 após limpeza (2.184 duplicadas; 9.862 de não alunos); 141 alunos com pré, pós e log. EdNet 89,27 mi linhas, 297.915 estudantes, 75.778 duplicadas | `ares/00_verbos.png`, `ednet/00_acoes_origens.png` |
| 3 | CRISP-DM e método: Spearman, controle do nível inicial, Holm; significância × magnitude | Tabela de leitura do ρ |
| 4 | Q1 | `comparacao/Q1_Q5_comparacao.png` (painel Q1) e `ednet/Q1_interacao_desempenho.png` |
| 5 | Q4 — mostrar **bruto × corrigido** (bom momento para explicar o rigor da análise) | `ares/Q4_atividades_engajamento.png`, `ednet/Q4_atividades_engajamento.png` |
| 6 | Q5 + fluxo 297.915 → 102.691 → 42.720 | `ednet/Q5_interacao_evolucao.png`, `ares/Q5_interacao_evolucao.png` |
| 7 | Q10 — frequência de uso × relação com o desempenho | `ares/Q10_...png`, `ednet/Q10_...png` |
| 8 | Comparação entre as bases | `comparacao/Q4_Q10_comparacao.png` |
| 9 | Recomendações por força da evidência (A/B/C) | Tabela da §9 |
| 10 | Limitações e o que a auditoria corrigiu | Tabela “Antes → Agora” (item 3 acima) |

**Para mostrar o passo a passo ao vivo:** cada slide de pergunta tem um notebook correspondente em `notebooks/`, já executado. Bons momentos para abrir o notebook durante a apresentação:
- `02_preparacao_ARES`, seção 1.3: a descoberta de que a nota oficial não conta a Q9 (71% → 100%).
- `04_Q1`, seção 3: a tabela que reproduz o 0,118 do notebook antigo.
- `05_Q4`, seção 1.2: a “atividade falsa” que todo aluno faz 4 vezes e que o método antigo marcava como −0,77.
- `06_Q5`, seção 3.2: 36% da janela inicial antiga era diagnóstico.

## 7. Perguntas prováveis da banca e como responder

**“Por que Spearman e não Pearson? O notebook de vocês dava 0,118.”**
São os mesmos 141 alunos. Com os mesmos dados, Pearson dá 0,115 e Spearman 0,079. O nº de interações é muito assimétrico (assimetria 1,9) e tem valores extremos, que puxam o Pearson. Spearman é o adequado. Os dois são não significativos (p > 0,15), então a conclusão é a mesma.

**“O EdNet tem 297.915 estudantes, mas a Q5 usa 42.720. Por quê?”**
São populações diferentes. Para medir evolução são precisas 20 questões de prática no início e 20 no fim. Só 102.691 estudantes responderam alguma questão fora do diagnóstico, e só 42.720 chegaram a 40. Cada análise usa o seu denominador, e isso está documentado em tabela (§7.3).

**“Se p < 0,001, por que vocês dizem que não há associação na Q5?”**
Com 42.720 estudantes, um ρ de 0,04 é estatisticamente distinguível de zero, mas explica cerca de 0,2% da variação. É significativo e desprezível ao mesmo tempo. Separamos significância de relevância prática.

**“No ARES não deu significativo. Então não existe relação?”**
Não dá para afirmar isso. Com 141 alunos, só detectaríamos ρ ≥ 0,24. O intervalo de confiança vai até +0,24 na Q1 e +0,17 na Q5, então um efeito fraco não pode ser descartado. Dizemos “não detectada”, não “inexistente”.

**“Por que controlar o volume de interações na Q4 e na Q10? Isso não remove o efeito que vocês querem medir?”**
Na Q4 o volume estava dos dois lados da correlação, o que gerava associações artificiais (ex.: o tutorial com −0,56 só porque é feito umas 3 vezes por todo mundo). Na Q10, controlar o volume separa “fazer esta atividade” de “usar mais a plataforma em geral”. Apresentamos as duas versões.

**“Por que excluir o diagnóstico do desempenho?”**
O diagnóstico é um nivelamento mais difícil (47% de acerto, contra 66% na prática). Quem usa pouco quase só fez o diagnóstico. Misturar as duas coisas fazia a frequência parecer associada ao desempenho (0,36), quando boa parte era só o tipo de questão (0,15 sem o diagnóstico).

**“O que é a correção de Holm e por que usá-la?”**
Testamos 10 atividades por base. Sem correção, a chance de pelo menos um falso positivo passa de 40%. Holm ajusta os p-valores para controlar isso. Foi o que derrubou o “vocabulário melhora o desempenho”.

**“Vocês podem recomendar mudanças no Beready com base em correlação?”**
Como hipóteses, sim; como certezas, não. Por isso cada recomendação tem a força da evidência e uma forma de testar (A/B). As que dependiam de resultados que não se sustentaram (feedback, recomendação adaptativa, mobile) foram retiradas.

**“Por que não usaram um modelo de machine learning?”**
As quatro perguntas são de **associação** (“existe relação…?”, “quais atividades estão associadas…?”), não de previsão. Correlação com controles e comparação de grupos responde diretamente a elas e é interpretável. Um modelo preditivo seria outra pergunta.

**“Os resultados valem para o Beready?”**
Valem como indício, principalmente onde as duas bases concordam (Q5 e “estudar conteúdo” na Q4). As populações são diferentes da do Beready, então a validação precisa ser feita com os dados do próprio site. Por isso recomendamos instrumentá-lo desde o início.

**“Onde está o passo a passo? Dá para ver o código?”**
Na pasta `notebooks/`: nove notebooks na ordem da documentação (01 entendimento → 09 auditoria), com código, tabelas e gráficos executados. Cada um termina conferindo os próprios números com os resultados oficiais.

**“Como garantem que os números do documento batem com o código?”**
O `auditoria.py` recalcula 93 resultados de forma independente, sem usar as funções da análise. Ele também confere se 21 números-chave aparecem na documentação. Se algo divergir, a execução termina com erro.
