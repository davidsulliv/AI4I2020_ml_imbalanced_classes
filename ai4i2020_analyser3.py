# -*- coding: utf-8 -*-
"""
TCC - Impacto do Desbalanceamento de Classes na Predição
de Falhas Industriais com Machine Learning

Versão revisada (v3) a partir do script original do autor.
Todas as alterações em relação ao script original estão comentadas
com a marca "# [MELHORIA]" para facilitar a comparação.

Modelos comparados (cada um com hiperparâmetros selecionados por
Grid Search com validação cruzada estratificada):
M1 - Regressão Logística
M2 - Regressão Logística Balanceada (class_weight="balanced")
M3 - Random Forest
M4 - Random Forest Balanceado (class_weight="balanced")
M5 - XGBoost
M6 - XGBoost Balanceado (scale_pos_weight)

Principais melhorias incorporadas nesta versão (ver relatório de
revisão em anexo para a justificativa detalhada de cada uma):

1. Caminhos relativos/portáveis via pathlib (o script original usava
   caminhos absolutos do Windows, o que impede a reprodução por
   terceiros e a own avaliação pela banca a partir de outro computador).
2. Checagem de integridade entre "Machine failure" e os indicadores de
   modo de falha (TWF/HDF/PWF/OSF/RNF) antes de descartá-los -- reforça
   com evidência quantitativa a decisão de excluí-los por vazamento de
   informação (data leakage), e documenta o achado de 27 registros
   (0,27%) em que os dois não coincidem, o que é uma característica
   conhecida da simulação da base AI4I 2020 (não é erro do código).
3. Regressão Logística também passa por Grid Search (parâmetro C),
   assim como Random Forest e XGBoost -- no script original apenas RF e
   XGBoost eram otimizados, o que tornava a comparação entre algoritmos
   desigual (RL era sempre avaliada nos hiperparâmetros padrão).
4. Otimização de limiar (threshold) por validação cruzada no treino,
   reportada lado a lado com o limiar padrão de 0,5 -- separa o que é
   "capacidade de ranquear" (ROC-AUC/PR-AUC, independentes de limiar)
   do que é "calibração do limiar" (Precision/Recall/F1 a 0,5), que é a
   causa da aparente vantagem da versão não balanceada do XGBoost.
5. Intervalos de confiança por bootstrap no conjunto de teste -- com
   apenas 68 falhas no teste, uma única divisão treino/teste produz
   estimativas pontuais com alta variância amostral; os IC quantificam
   essa incerteza.
6. Métricas adicionais recomendadas para dados desbalanceados:
   Balanced Accuracy e MCC (Matthews Correlation Coefficient), que não
   saturam com o desbalanceamento como a Accuracy simples.
7. Registro das versões das bibliotecas utilizadas (reprodutibilidade).
8. Redução de duplicação de código por meio de funções auxiliares
   (avaliação de modelo, salvamento de figuras, extração de
   importâncias/coeficientes).
9. Modelos finais salvos em disco (joblib) para reuso/implantação.

Observação: o código mantém a mesma estrutura conceitual e os mesmos
seis modelos do script original -- nada foi removido, apenas
adicionado/corrigido -- para que os resultados continuem comparáveis
com o restante do TCC.
"""

# %%
# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import xgboost
from joblib import dump

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# ============================================================
# 2. CONFIGURAÇÕES
# ============================================================
# [MELHORIA] Caminhos relativos ao próprio script (funciona em
# qualquer computador/SO, sem editar nada). Pode ser sobrescrito por
# variável de ambiente, se necessário rodar com outra base.

PASTA_BASE = Path(__file__).resolve().parent
CAMINHO_ARQUIVO = Path(
    __import__("os").environ.get(
        "AI4I_CSV", PASTA_BASE / "ai4i2020.csv"
    )
)
PASTA_RESULTADOS = PASTA_BASE / "Resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)
PASTA_MODELOS = PASTA_RESULTADOS / "modelos"
PASTA_MODELOS.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 1000  # [MELHORIA] nº de reamostragens para IC no teste

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


def salvar_figura(fig, nome_arquivo):
    """[MELHORIA] Centraliza o salvamento de figuras (evita repetir
    os mesmos três parâmetros em ~15 lugares diferentes)."""
    fig.tight_layout()
    fig.savefig(
        PASTA_RESULTADOS / nome_arquivo,
        bbox_inches="tight",
    )
    plt.close(fig)


# [MELHORIA] Registro das versões usadas, para reprodutibilidade
# (inclua isto no apêndice metodológico do TCC).
versoes = {
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
}

print("\n========================================")
print("VERSÕES DAS BIBLIOTECAS (reprodutibilidade)")
print("========================================")
for pacote, versao in versoes.items():
    print(f"{pacote}: {versao}")

with open(PASTA_RESULTADOS / "versoes_bibliotecas.json", "w") as f:
    json.dump(versoes, f, indent=2)

# %%
# ============================================================
# 3. LEITURA DA BASE
# ============================================================

df = pd.read_csv(CAMINHO_ARQUIVO)

print("\n========================================")
print("DIMENSÃO DA BASE")
print("========================================")
print(df.shape)

print("\n========================================")
print("VALORES AUSENTES")
print("========================================")
print(df.isnull().sum())

# %%
# ============================================================
# 4. CHECAGEM DE INTEGRIDADE DOS INDICADORES DE FALHA
# ============================================================
# [MELHORIA] O script original já excluía TWF/HDF/PWF/OSF/RNF da
# modelagem (decisão correta, evita vazamento de informação), mas não
# verificava, com números, o quanto "Machine failure" de fato coincide
# com a união desses indicadores. Essa checagem sustenta com evidência
# a justificativa de exclusão descrita no texto do TCC e documenta uma
# característica da base que vale citar na seção de limitações.

colunas_flag = ["TWF", "HDF", "PWF", "OSF", "RNF"]
uniao_flags = df[colunas_flag].max(axis=1)
divergencias = (df["Machine failure"] != uniao_flags).sum()

print("\n========================================")
print("INTEGRIDADE: Machine failure vs. indicadores de falha")
print("========================================")
print(
    f"Registros em que 'Machine failure' diverge da união dos "
    f"indicadores TWF/HDF/PWF/OSF/RNF: {divergencias} "
    f"({divergencias / len(df):.2%} da base)."
)
print(
    "Interpretação: a documentação da base AI4I 2020 descreve que o "
    "indicador RNF pode ocorrer sem gerar falha registrada (falha "
    "aleatória de baixa probabilidade) e que nem toda falha registrada "
    "está associada a um dos cinco modos catalogados. A divergência "
    "confirma que os indicadores não são um espelho perfeito do alvo, "
    "mas isso não anula o risco de vazamento: mesmo imperfeitos, eles "
    "carregam informação diretamente derivada da definição do alvo e "
    "por isso permanecem corretamente excluídos da modelagem."
)

# %%
# ============================================================
# 5. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

variaveis = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

alvo = "Machine failure"

variaveis_categoricas = ["Type"]
variaveis_numericas = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# %%
# ============================================================
# 6. DISTRIBUIÇÃO DA VARIÁVEL ALVO
# ============================================================

quantidade_classes = df[alvo].value_counts().sort_index()
percentual_classes = df[alvo].value_counts(normalize=True).sort_index() * 100

print("\n========================================")
print("DISTRIBUIÇÃO DAS CLASSES")
print("========================================")
for classe in quantidade_classes.index:
    print(
        f"Classe {classe}: {quantidade_classes[classe]} registros "
        f"({percentual_classes[classe]:.2f}%)"
    )

fig, ax = plt.subplots(figsize=(7, 5))
rotulos = ["Sem falha", "Falha"]
valores = [quantidade_classes.get(0, 0), quantidade_classes.get(1, 0)]
barras = ax.bar(rotulos, valores)
ax.set_title("Distribuição da variável Machine failure")
ax.set_ylabel("Número de observações")
for barra, valor in zip(barras, valores):
    percentual = (valor / len(df)) * 100
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor}\n({percentual:.2f}%)",
        ha="center",
        va="bottom",
    )
salvar_figura(fig, "01_distribuicao_classes.png")

# %%
# ============================================================
# 7. DISTRIBUIÇÃO POR TIPO DE PRODUTO
# ============================================================

tabela_tipo = pd.crosstab(df["Type"], df[alvo])

fig, ax = plt.subplots(figsize=(8, 5))
tabela_tipo.plot(kind="bar", ax=ax)
ax.set_title("Distribuição de falhas por tipo de produto")
ax.set_xlabel("Tipo do produto")
ax.set_ylabel("Número de observações")
ax.legend(["Sem falha", "Falha"])
plt.setp(ax.get_xticklabels(), rotation=0)
salvar_figura(fig, "02_falhas_por_tipo.png")
# [MELHORIA] Esta figura já era gerada no script original, mas não
# possuía referência/legenda no texto do TCC -- sugerimos incluir
# como Figura no corpo do trabalho (ver relatório de revisão), pois a
# taxa de falha varia por tipo (H: 2,09%; L: 3,92%; M: 2,77%).

# %%
# ============================================================
# 8. ESTATÍSTICAS DESCRITIVAS E BOXPLOTS
# ============================================================

estatisticas = df[variaveis_numericas].describe().T
estatisticas.to_excel(PASTA_RESULTADOS / "03_estatisticas_descritivas.xlsx")

for indice, coluna in enumerate(variaveis_numericas, start=1):
    dados_sem_falha = df.loc[df[alvo] == 0, coluna]
    dados_com_falha = df.loc[df[alvo] == 1, coluna]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [dados_sem_falha, dados_com_falha],
        tick_labels=["Sem falha", "Falha"],
    )
    ax.set_title(f"{coluna} por ocorrência de falha")
    ax.set_ylabel(coluna)
    nome = (
        coluna.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
    )
    salvar_figura(fig, f"04_{indice}_boxplot_{nome}.png")

# %%
# ============================================================
# 9. MATRIZ DE CORRELAÇÃO
# ============================================================

dados_correlacao = df[variaveis_numericas + [alvo]]
correlacao = dados_correlacao.corr()

fig, ax = plt.subplots(figsize=(9, 7))
imagem = ax.imshow(correlacao.values, aspect="auto")
ax.set_xticks(range(len(correlacao.columns)))
ax.set_xticklabels(correlacao.columns, rotation=45, ha="right")
ax.set_yticks(range(len(correlacao.index)))
ax.set_yticklabels(correlacao.index)
for i in range(len(correlacao.index)):
    for j in range(len(correlacao.columns)):
        ax.text(j, i, f"{correlacao.iloc[i, j]:.2f}", ha="center", va="center")
plt.colorbar(imagem, ax=ax)
ax.set_title("Matriz de correlação das variáveis")
salvar_figura(fig, "05_matriz_correlacao.png")

# %%
# ============================================================
# 10. SEPARAÇÃO X/Y E DIVISÃO TREINO/TESTE
# ============================================================

X = df[variaveis].copy()
y = df[alvo].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print("\n========================================")
print("DIVISÃO TREINO / TESTE")
print("========================================")
print("Treinamento:", len(X_train), " | Teste:", len(X_test))
print("Falhas no treinamento (%):", round(y_train.mean() * 100, 2))
print("Falhas no teste (%):", round(y_test.mean() * 100, 2))
print("Falhas absolutas no teste:", int(y_test.sum()))
print(
    "[Nota metodológica] Com apenas",
    int(y_test.sum()),
    "casos positivos no teste, cada acerto/erro adicional altera o "
    "Recall em ~1,5 p.p. -- ver seção de intervalos de confiança "
    "(bootstrap) mais adiante para dimensionar essa incerteza.",
)

# %%
# ============================================================
# 11. PRÉ-PROCESSAMENTO
# ============================================================

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "categorica",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            variaveis_categoricas,
        ),
        ("numerica", StandardScaler(), variaveis_numericas),
    ]
)

validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %%
# ============================================================
# 12. FUNÇÃO AUXILIAR DE GRID SEARCH
# ============================================================
# [MELHORIA] Uma única função treina qualquer um dos três algoritmos,
# eliminando ~150 linhas duplicadas do script original.


def ajustar_grid(nome, estimador, grade_parametros, X_tr, y_tr):
    pipeline = Pipeline(
        steps=[("preprocessamento", clone(preprocessamento)), ("modelo", estimador)]
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=grade_parametros,
        scoring="average_precision",
        cv=validacao,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr, y_tr)
    print(f"\n[{nome}] melhores parâmetros: {grid.best_params_}")
    print(f"[{nome}] PR-AUC (CV): {grid.best_score_:.4f}")
    return grid


# %%
# ============================================================
# 13. REGRESSÃO LOGÍSTICA (SEM E COM BALANCEAMENTO)
# ============================================================
# [MELHORIA] No script original a Regressão Logística usava sempre os
# hiperparâmetros padrão do scikit-learn, enquanto Random Forest e
# XGBoost passavam por Grid Search. Isso favorecia os modelos em
# árvore na comparação. Agora a RL também é otimizada (parâmetro de
# regularização C), tornando a comparação mais justa entre os três
# algoritmos.

grade_rl = {"modelo__C": [0.01, 0.1, 1, 10, 100]}

grid_rl = ajustar_grid(
    "Regressão Logística",
    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    grade_rl,
    X_train,
    y_train,
)
modelo_rl = grid_rl.best_estimator_

grid_rl_balanceado = ajustar_grid(
    "Regressão Logística Balanceada",
    LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    grade_rl,
    X_train,
    y_train,
)
modelo_rl_balanceado = grid_rl_balanceado.best_estimator_

# %%
# ============================================================
# 14. RANDOM FOREST (SEM E COM BALANCEAMENTO)
# ============================================================

grade_rf = {
    "modelo__n_estimators": [100, 200],
    "modelo__max_depth": [None, 5, 10],
    "modelo__min_samples_split": [2, 5],
}

grid_rf = ajustar_grid(
    "Random Forest",
    RandomForestClassifier(random_state=RANDOM_STATE),
    grade_rf,
    X_train,
    y_train,
)
modelo_rf = grid_rf.best_estimator_

grid_rf_balanceado = ajustar_grid(
    "Random Forest Balanceado",
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    grade_rf,
    X_train,
    y_train,
)
modelo_rf_balanceado = grid_rf_balanceado.best_estimator_

# %%
# ============================================================
# 15. XGBOOST (SEM E COM BALANCEAMENTO)
# ============================================================

grade_xgb = {
    "modelo__n_estimators": [100, 200],
    "modelo__max_depth": [3, 5],
    "modelo__learning_rate": [0.05, 0.10],
}

grid_xgb = ajustar_grid(
    "XGBoost",
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # [MELHORIA] evita não-determinismo por paralelismo interno
    ),
    grade_xgb,
    X_train,
    y_train,
)
modelo_xgb = grid_xgb.best_estimator_

n_negativos = (y_train == 0).sum()
n_positivos = (y_train == 1).sum()
scale_pos_weight = n_negativos / n_positivos
print(f"\nscale_pos_weight (XGBoost balanceado) = {scale_pos_weight:.4f}")

grid_xgb_balanceado = ajustar_grid(
    "XGBoost Balanceado",
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
    grade_xgb,
    X_train,
    y_train,
)
modelo_xgb_balanceado = grid_xgb_balanceado.best_estimator_

# [MELHORIA] Nota de reprodutibilidade a incluir no TCC: mesmo com
# random_state fixo, o XGBoost pode produzir resultados ligeiramente
# diferentes (poucas unidades de acerto/erro em ~2.000 casos de teste)
# entre versões da biblioteca ou entre execuções com paralelismo
# interno habilitado, pois o algoritmo histogram-based faz operações
# de ponto flutuante cuja ordem pode variar entre threads. Fixamos
# n_jobs=1 no estimador para reduzir essa fonte de variação; ainda
# assim, recomenda-se registrar a versão exata do xgboost usada (ver
# arquivo versoes_bibliotecas.json) e tratar pequenas diferenças de
# 1-2 casos como esperadas, não como erro.

modelos = {
    "M1 - Regressão Logística": modelo_rl,
    "M2 - Regressão Logística Balanceada": modelo_rl_balanceado,
    "M3 - Random Forest": modelo_rf,
    "M4 - Random Forest Balanceado": modelo_rf_balanceado,
    "M5 - XGBoost": modelo_xgb,
    "M6 - XGBoost Balanceado": modelo_xgb_balanceado,
}

for nome_modelo, modelo in modelos.items():
    dump(modelo, PASTA_MODELOS / f"{nome_modelo[:2]}.joblib")  # [MELHORIA]

# %%
# ============================================================
# 16. LIMIAR ÓTIMO POR VALIDAÇÃO CRUZADA NO TREINO
# ============================================================
# [MELHORIA] Todo o script original avaliava os modelos apenas no
# limiar padrão de 0,5. Isso confunde duas coisas: a capacidade do
# modelo de ranquear corretamente os casos (medida por ROC-AUC/PR-AUC,
# que não dependem de limiar) e a adequação do limiar de decisão
# (da qual dependem Precision/Recall/F1). Modelos com class_weight ou
# scale_pos_weight deslocam as probabilidades previstas, então 0,5
# deixa de ser um limiar razoável para eles -- o que explica boa parte
# da "vantagem" aparente das versões não balanceadas nas métricas a
# 0,5. Aqui, o limiar que maximiza o F1 é escolhido por validação
# cruzada DENTRO do treino (nunca olhando o teste) e depois aplicado
# ao conjunto de teste.


def limiar_otimo_cv(pipeline_ajustado, X_tr, y_tr, cv):
    """Obtém probabilidades fora-da-dobra (out-of-fold) no treino via
    cross_val_predict e retorna o limiar que maximiza o F1."""
    probas_oof = cross_val_predict(
        clone(pipeline_ajustado), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    precisoes, recalls, limiares = precision_recall_curve(y_tr, probas_oof)
    f1s = np.divide(
        2 * precisoes * recalls,
        precisoes + recalls,
        out=np.zeros_like(precisoes),
        where=(precisoes + recalls) != 0,
    )
    melhor_indice = np.argmax(f1s[:-1])  # último ponto não tem limiar associado
    return float(limiares[melhor_indice])


limiares_otimos = {}
for nome_modelo, modelo in modelos.items():
    limiares_otimos[nome_modelo] = limiar_otimo_cv(modelo, X_train, y_train, validacao)

print("\n========================================")
print("LIMIARES ÓTIMOS (F1) POR VALIDAÇÃO CRUZADA NO TREINO")
print("========================================")
for nome_modelo, limiar in limiares_otimos.items():
    print(f"{nome_modelo}: {limiar:.3f}")

# %%
# ============================================================
# 17. AVALIAÇÃO NO CONJUNTO DE TESTE (LIMIAR PADRÃO E ÓTIMO)
# ============================================================


def calcular_metricas(y_verdadeiro, y_predito, y_prob):
    matriz = confusion_matrix(y_verdadeiro, y_predito)
    tn, fp, fn, tp = matriz.ravel()
    return {
        "Accuracy": accuracy_score(y_verdadeiro, y_predito),
        "Balanced_Accuracy": balanced_accuracy_score(y_verdadeiro, y_predito),
        "Precision": precision_score(y_verdadeiro, y_predito, zero_division=0),
        "Recall": recall_score(y_verdadeiro, y_predito, zero_division=0),
        "F1": f1_score(y_verdadeiro, y_predito, zero_division=0),
        "MCC": matthews_corrcoef(y_verdadeiro, y_predito),
        "ROC-AUC": roc_auc_score(y_verdadeiro, y_prob),
        "PR-AUC": average_precision_score(y_verdadeiro, y_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
    # [MELHORIA] Balanced Accuracy e MCC foram adicionadas por serem
    # recomendadas na literatura de classificação desbalanceada
    # (MCC, em particular, resume a matriz de confusão inteira em um
    # único número que não satura mesmo com desbalanceamento extremo).


resultados_padrao = []
resultados_otimo = []
dados_curvas = {}
probas_teste = {}

for nome_modelo, modelo in modelos.items():
    probabilidade = modelo.predict_proba(X_test)[:, 1]
    probas_teste[nome_modelo] = probabilidade

    previsao_padrao = modelo.predict(X_test)  # limiar 0,5 (comportamento original)
    metricas_padrao = calcular_metricas(y_test, previsao_padrao, probabilidade)
    metricas_padrao["Modelo"] = nome_modelo
    metricas_padrao["Limiar"] = 0.5
    resultados_padrao.append(metricas_padrao)

    limiar = limiares_otimos[nome_modelo]
    previsao_otima = (probabilidade >= limiar).astype(int)
    metricas_otimas = calcular_metricas(y_test, previsao_otima, probabilidade)
    metricas_otimas["Modelo"] = nome_modelo
    metricas_otimas["Limiar"] = limiar
    resultados_otimo.append(metricas_otimas)

    fpr, tpr, _ = roc_curve(y_test, probabilidade)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilidade)
    dados_curvas[nome_modelo] = {
        "fpr": fpr, "tpr": tpr, "precision": precision_curve, "recall": recall_curve,
    }

    print(f"\n{'=' * 40}\n{nome_modelo} (limiar padrão 0,5)\n{'=' * 40}")
    print(classification_report(y_test, previsao_padrao, digits=4))

colunas_ordenadas = [
    "Modelo", "Limiar", "Accuracy", "Balanced_Accuracy", "Precision",
    "Recall", "F1", "MCC", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP",
]
df_resultados = pd.DataFrame(resultados_padrao)[colunas_ordenadas]
df_resultados_otimo = pd.DataFrame(resultados_otimo)[colunas_ordenadas]

print("\n========================================")
print("RESULTADOS - LIMIAR PADRÃO (0,5)")
print("========================================")
print(df_resultados.round(4))

print("\n========================================")
print("RESULTADOS - LIMIAR OTIMIZADO POR CV NO TREINO")
print("========================================")
print(df_resultados_otimo.round(4))

# %%
# ============================================================
# 18. INTERVALOS DE CONFIANÇA POR BOOTSTRAP NO TESTE
# ============================================================
# [MELHORIA] Quantifica a incerteza amostral de uma única divisão
# treino/teste com apenas 68 casos positivos. Reamostra o conjunto de
# teste (com reposição) N_BOOTSTRAP vezes e recalcula as métricas a
# partir das probabilidades já previstas -- não há retreinamento, então
# o custo computacional é desprezível.

rng = np.random.default_rng(RANDOM_STATE)
y_test_arr = y_test.to_numpy()
n_teste = len(y_test_arr)

linhas_ic = []
for nome_modelo in modelos:
    prob = probas_teste[nome_modelo]
    pred_padrao = (prob >= 0.5).astype(int)

    amostras = {"Recall": [], "Precision": [], "F1": [], "ROC-AUC": [], "PR-AUC": []}
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n_teste, n_teste)
        y_b, pred_b, prob_b = y_test_arr[indices], pred_padrao[indices], prob[indices]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue  # reamostra sem as duas classes: descarta esta iteração
        amostras["Recall"].append(recall_score(y_b, pred_b, zero_division=0))
        amostras["Precision"].append(precision_score(y_b, pred_b, zero_division=0))
        amostras["F1"].append(f1_score(y_b, pred_b, zero_division=0))
        amostras["ROC-AUC"].append(roc_auc_score(y_b, prob_b))
        amostras["PR-AUC"].append(average_precision_score(y_b, prob_b))

    linha = {"Modelo": nome_modelo}
    for metrica, valores in amostras.items():
        linha[f"{metrica}_p2.5"] = np.percentile(valores, 2.5)
        linha[f"{metrica}_p50"] = np.percentile(valores, 50)
        linha[f"{metrica}_p97.5"] = np.percentile(valores, 97.5)
    linhas_ic.append(linha)

df_ic_bootstrap = pd.DataFrame(linhas_ic)

print("\n========================================")
print(f"INTERVALOS DE CONFIANÇA (95%) POR BOOTSTRAP - n={N_BOOTSTRAP}")
print("========================================")
print(
    df_ic_bootstrap[
        ["Modelo", "Recall_p2.5", "Recall_p50", "Recall_p97.5",
         "F1_p2.5", "F1_p50", "F1_p97.5"]
    ].round(3)
)

# %%
# ============================================================
# 19. SALVAR RESULTADOS EM EXCEL
# ============================================================

arquivo_excel = PASTA_RESULTADOS / "Resultados_Modelos_AI4I.xlsx"
with pd.ExcelWriter(arquivo_excel, engine="openpyxl") as writer:
    df_resultados.to_excel(writer, sheet_name="Metricas_limiar_05", index=False)
    df_resultados_otimo.to_excel(writer, sheet_name="Metricas_limiar_otimo", index=False)
    df_ic_bootstrap.to_excel(writer, sheet_name="IC_Bootstrap_95", index=False)
    estatisticas.to_excel(writer, sheet_name="Estatisticas")
    correlacao.to_excel(writer, sheet_name="Correlacao")
    tabela_tipo.to_excel(writer, sheet_name="Falhas_Tipo")

print("\nArquivo Excel salvo em:", arquivo_excel)

# %%
# ============================================================
# 20. MATRIZES DE CONFUSÃO (CONTAGEM E NORMALIZADA)
# ============================================================
# [MELHORIA] Adicionada também a versão normalizada por linha (taxa),
# pois em bases muito desbalanceadas a matriz em contagem absoluta é
# dominada visualmente pelos verdadeiros negativos.

for numero, (nome_modelo, modelo) in enumerate(modelos.items(), start=1):
    previsao = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, previsao)
    matriz_normalizada = confusion_matrix(y_test, previsao, normalize="true")

    fig, eixos = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dados, titulo, fmt in [
        (eixos[0], matriz, "Contagem", "d"),
        (eixos[1], matriz_normalizada, "Normalizada por linha", ".2f"),
    ]:
        imagem = ax.imshow(dados)
        for i in range(2):
            for j in range(2):
                valor = dados[i, j]
                texto = f"{valor:d}" if fmt == "d" else f"{valor:.2f}"
                ax.text(j, i, texto, ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Sem falha", "Falha"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Sem falha", "Falha"])
        ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe real")
        ax.set_title(titulo)
    fig.suptitle(f"Matriz de confusão - {nome_modelo}")
    salvar_figura(fig, f"06_{numero}_matriz_confusao.png")

# %%
# ============================================================
# 21. CURVAS ROC E PRECISION-RECALL
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))
for nome_modelo, dados in dados_curvas.items():
    valor_auc = df_resultados.loc[df_resultados["Modelo"] == nome_modelo, "ROC-AUC"].iloc[0]
    ax.plot(dados["fpr"], dados["tpr"], label=f"{nome_modelo} (AUC={valor_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("Taxa de falsos positivos"); ax.set_ylabel("Taxa de verdadeiros positivos")
ax.set_title("Curvas ROC dos modelos"); ax.legend(fontsize=8)
salvar_figura(fig, "07_curvas_roc.png")

fig, ax = plt.subplots(figsize=(9, 7))
for nome_modelo, dados in dados_curvas.items():
    valor_pr = df_resultados.loc[df_resultados["Modelo"] == nome_modelo, "PR-AUC"].iloc[0]
    ax.plot(dados["recall"], dados["precision"], label=f"{nome_modelo} (PR-AUC={valor_pr:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Curvas Precision-Recall"); ax.legend(fontsize=8)
salvar_figura(fig, "08_curvas_precision_recall.png")

# %%
# ============================================================
# 22. COMPARAÇÕES GRÁFICAS DE MÉTRICAS
# ============================================================

metricas_grafico = df_resultados.set_index("Modelo")[
    ["Accuracy", "Precision", "Recall", "F1", "MCC"]
]
fig, ax = plt.subplots(figsize=(11, 6))
metricas_grafico.plot(kind="bar", ax=ax)
ax.set_title("Comparação do desempenho dos modelos (limiar 0,5)")
ax.set_ylabel("Valor da métrica"); ax.set_ylim(-0.2, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
salvar_figura(fig, "09_comparacao_metricas.png")

fig, ax = plt.subplots(figsize=(10, 6))
df_resultados.set_index("Modelo")[["Recall", "F1"]].plot(kind="bar", ax=ax)
ax.set_title("Comparação de Recall e F1-score"); ax.set_ylabel("Valor")
ax.set_ylim(0, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
salvar_figura(fig, "10_recall_f1.png")

# %%
# ============================================================
# 23. IMPORTÂNCIA DE VARIÁVEIS / COEFICIENTES
# ============================================================
# [MELHORIA] Função única substitui os três blocos quase idênticos do
# script original (RF, XGBoost e RL).


def extrair_importancia(pipeline_ajustado):
    preprocessador = pipeline_ajustado.named_steps["preprocessamento"]
    modelo_final = pipeline_ajustado.named_steps["modelo"]
    nomes = preprocessador.get_feature_names_out()
    if hasattr(modelo_final, "feature_importances_"):
        valores = modelo_final.feature_importances_
    else:
        valores = modelo_final.coef_[0]
    return pd.Series(valores, index=nomes).sort_values()


for nome_modelo, nome_arquivo, titulo in [
    ("M4 - Random Forest Balanceado", "11_importancia_variaveis_random_forest.png",
     "Importância das variáveis no Random Forest balanceado"),
    ("M6 - XGBoost Balanceado", "14_importancia_variaveis_xgboost.png",
     "Importância das variáveis no XGBoost balanceado"),
]:
    importancias = extrair_importancia(modelos[nome_modelo])
    fig, ax = plt.subplots(figsize=(9, 6))
    importancias.plot(kind="barh", ax=ax)
    ax.set_title(titulo); ax.set_xlabel("Importância")
    salvar_figura(fig, nome_arquivo)

coeficientes = extrair_importancia(modelos["M2 - Regressão Logística Balanceada"])
fig, ax = plt.subplots(figsize=(9, 6))
coeficientes.plot(kind="barh", ax=ax)
ax.axvline(0)
ax.set_title("Coeficientes da Regressão Logística balanceada")
ax.set_xlabel("Coeficiente")
salvar_figura(fig, "12_coeficientes_regressao_logistica.png")

# %%
# ============================================================
# 24. GRID SEARCH - RESUMO
# ============================================================

resultados_grid = pd.DataFrame({
    "Modelo": [
        "Regressão Logística", "Regressão Logística Balanceada",
        "Random Forest", "Random Forest Balanceado",
        "XGBoost", "XGBoost Balanceado",
    ],
    "Melhores parâmetros": [
        str(grid_rl.best_params_), str(grid_rl_balanceado.best_params_),
        str(grid_rf.best_params_), str(grid_rf_balanceado.best_params_),
        str(grid_xgb.best_params_), str(grid_xgb_balanceado.best_params_),
    ],
    "PR-AUC validação cruzada": [
        grid_rl.best_score_, grid_rl_balanceado.best_score_,
        grid_rf.best_score_, grid_rf_balanceado.best_score_,
        grid_xgb.best_score_, grid_xgb_balanceado.best_score_,
    ],
})
resultados_grid.to_excel(PASTA_RESULTADOS / "13_resultados_grid_search.xlsx", index=False)

print("\n" + "=" * 70)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 70)
print("\nResultados salvos em:", PASTA_RESULTADOS)
