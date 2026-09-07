# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:54:23 2026

@author: 693davidson
"""

# -*- coding: utf-8 -*-

"""
TCC - Impacto do Desbalanceamento de Classes na Predição
de Falhas Industriais com Machine Learning

Modelos:
M1 - Regressão Logística
M2 - Regressão Logística Balanceada
M3 - Random Forest
M4 - Random Forest Balanceado
"""
#%%
# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier
#%%
# ============================================================
# 2. CONFIGURAÇÕES
# ============================================================

CAMINHO_ARQUIVO = r"C:\TCC\ai4i2020.csv"

PASTA_RESULTADOS = r"C:\TCC\Resultados"

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

RANDOM_STATE = 42

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

#%%
# ============================================================
# 3. LEITURA DA BASE
# ============================================================

df = pd.read_csv(CAMINHO_ARQUIVO)

print("\n========================================")
print("DIMENSÃO DA BASE")
print("========================================")

print(df.shape)

print("\n========================================")
print("COLUNAS")
print("========================================")

print(df.columns.tolist())

print("\n========================================")
print("CONFERÊNCIA DAS TEMPERATURAS")
print("========================================")

print(
    df[
        [
            "Air temperature [K]",
            "Process temperature [K]"
        ]
    ].head(10)
)

print("\n========================================")
print("ESTATÍSTICAS DAS TEMPERATURAS")
print("========================================")

print(
    df[
        [
            "Air temperature [K]",
            "Process temperature [K]"
        ]
    ].describe()
)

print("\n========================================")
print("PRIMEIROS REGISTROS")
print("========================================")

print(df.head())

print("\n========================================")
print("VALORES AUSENTES")
print("========================================")

print(df.isnull().sum())

#%%
# ============================================================
# 4. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

variaveis = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

alvo = "Machine failure"

#%%
# ============================================================
# 5. DISTRIBUIÇÃO DA VARIÁVEL ALVO
# ============================================================

quantidade_classes = df[alvo].value_counts().sort_index()

percentual_classes = (
    df[alvo]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\n========================================")
print("DISTRIBUIÇÃO DAS CLASSES")
print("========================================")

for classe in quantidade_classes.index:

    print(
        f"Classe {classe}: "
        f"{quantidade_classes[classe]} registros "
        f"({percentual_classes[classe]:.2f}%)"
    )

#%%
# ============================================================
# 6. GRÁFICO - DISTRIBUIÇÃO DAS CLASSES
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

rotulos = [
    "Sem falha",
    "Falha"
]

valores = [
    quantidade_classes.get(0, 0),
    quantidade_classes.get(1, 0)
]

barras = ax.bar(rotulos, valores)

ax.set_title(
    "Distribuição da variável Machine failure"
)

ax.set_ylabel(
    "Número de observações"
)

for barra, valor in zip(barras, valores):

    percentual = (
        valor / len(df)
    ) * 100

    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor}\n({percentual:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "01_distribuicao_classes.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 7. DISTRIBUIÇÃO POR TIPO DE PRODUTO
# ============================================================

tabela_tipo = pd.crosstab(
    df["Type"],
    df[alvo]
)

print("\n========================================")
print("FALHAS POR TIPO")
print("========================================")

print(tabela_tipo)


tabela_tipo.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Distribuição de falhas por tipo de produto"
)

plt.xlabel(
    "Tipo do produto"
)

plt.ylabel(
    "Número de observações"
)

plt.legend(
    [
        "Sem falha",
        "Falha"
    ]
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "02_falhas_por_tipo.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 8. ESTATÍSTICAS DESCRITIVAS
# ============================================================

variaveis_numericas = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

estatisticas = (
    df[variaveis_numericas]
    .describe()
    .T
)

print("\n========================================")
print("ESTATÍSTICAS DESCRITIVAS")
print("========================================")

print(estatisticas)


estatisticas.to_excel(
    os.path.join(
        PASTA_RESULTADOS,
        "03_estatisticas_descritivas.xlsx"
    )
)

#%%
# ============================================================
# 9. BOXPLOTS
# ============================================================

for indice, coluna in enumerate(
    variaveis_numericas,
    start=1
):

    dados_sem_falha = df.loc[
        df[alvo] == 0,
        coluna
    ]

    dados_com_falha = df.loc[
        df[alvo] == 1,
        coluna
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        [
            dados_sem_falha,
            dados_com_falha
        ],
        tick_labels=[
            "Sem falha",
            "Falha"
        ]
    )

    ax.set_title(
        f"{coluna} por ocorrência de falha"
    )

    ax.set_ylabel(
        coluna
    )

    plt.tight_layout()

    nome = (
        coluna
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
    )

    plt.savefig(
        os.path.join(
            PASTA_RESULTADOS,
            f"04_{indice}_boxplot_{nome}.png"
        ),
        bbox_inches="tight"
    )

    plt.show()

#%%
# ============================================================
# 10. MATRIZ DE CORRELAÇÃO
# ============================================================

dados_correlacao = df[
    variaveis_numericas + [alvo]
]

correlacao = (
    dados_correlacao
    .corr()
)

print("\n========================================")
print("MATRIZ DE CORRELAÇÃO")
print("========================================")

print(
    correlacao.round(3)
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)

imagem = ax.imshow(
    correlacao.values,
    aspect="auto"
)

ax.set_xticks(
    range(
        len(correlacao.columns)
    )
)

ax.set_xticklabels(
    correlacao.columns,
    rotation=45,
    ha="right"
)

ax.set_yticks(
    range(
        len(correlacao.index)
    )
)

ax.set_yticklabels(
    correlacao.index
)


for i in range(
    len(correlacao.index)
):

    for j in range(
        len(correlacao.columns)
    ):

        ax.text(
            j,
            i,
            f"{correlacao.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )


plt.colorbar(
    imagem,
    ax=ax
)

plt.title(
    "Matriz de correlação das variáveis"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "05_matriz_correlacao.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 11. SEPARAÇÃO ENTRE X E Y
# ============================================================

X = df[variaveis].copy()

y = df[alvo].copy()

#%%
# ============================================================
# 12. DIVISÃO TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )
)


print("\n========================================")
print("DIVISÃO TREINO / TESTE")
print("========================================")

print(
    "Treinamento:",
    len(X_train)
)

print(
    "Teste:",
    len(X_test)
)


print(
    "\nFalhas no treinamento:"
)

print(
    y_train
    .value_counts(normalize=True)
    .sort_index()
)


print(
    "\nFalhas no teste:"
)

print(
    y_test
    .value_counts(normalize=True)
    .sort_index()
)

#%%
# ============================================================
# 13. PRÉ-PROCESSAMENTO
# ============================================================

variaveis_categoricas = [
    "Type"
]

variaveis_numericas_modelo = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


preprocessamento = (
    ColumnTransformer(
        transformers=[
            (
                "categorica",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                variaveis_categoricas
            ),

            (
                "numerica",
                StandardScaler(),
                variaveis_numericas_modelo
            )
        ]
    )
)

#%%
# ============================================================
# 14. REGRESSÃO LOGÍSTICA - SEM BALANCEAMENTO
# ============================================================

modelo_rl = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),

        (
            "modelo",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


modelo_rl.fit(
    X_train,
    y_train
)

#%%
# ============================================================
# 15. REGRESSÃO LOGÍSTICA - BALANCEADA
# ============================================================

modelo_rl_balanceado = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),

        (
            "modelo",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


modelo_rl_balanceado.fit(
    X_train,
    y_train
)

#%%
# ============================================================
# 16. RANDOM FOREST - GRID SEARCH
# ============================================================

pipeline_rf = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),

        (
            "modelo",
            RandomForestClassifier(
                random_state=RANDOM_STATE
            )
        )
    ]
)


grade_parametros = {

    "modelo__n_estimators": [
        100,
        200
    ],

    "modelo__max_depth": [
        None,
        5,
        10
    ],

    "modelo__min_samples_split": [
        2,
        5
    ]

}


validacao = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


grid_rf = GridSearchCV(

    estimator=pipeline_rf,

    param_grid=grade_parametros,

    scoring="average_precision",

    cv=validacao,

    n_jobs=-1,

    verbose=1
)


grid_rf.fit(
    X_train,
    y_train
)


modelo_rf = (
    grid_rf.best_estimator_
)


print("\n========================================")
print("MELHOR RANDOM FOREST")
print("========================================")

print(
    grid_rf.best_params_
)

print(
    "PR-AUC CV:",
    grid_rf.best_score_
)

#%%
# ============================================================
# 17. RANDOM FOREST BALANCEADO
# ============================================================

pipeline_rf_balanceado = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),

        (
            "modelo",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


grid_rf_balanceado = GridSearchCV(

    estimator=pipeline_rf_balanceado,

    param_grid=grade_parametros,

    scoring="average_precision",

    cv=validacao,

    n_jobs=-1,

    verbose=1
)


grid_rf_balanceado.fit(
    X_train,
    y_train
)


modelo_rf_balanceado = (
    grid_rf_balanceado.best_estimator_
)


print("\n========================================")
print("MELHOR RANDOM FOREST BALANCEADO")
print("========================================")

print(
    grid_rf_balanceado.best_params_
)

print(
    "PR-AUC CV:",
    grid_rf_balanceado.best_score_
)

#%%
# ============================================================
# 18. XGBOOST - SEM BALANCEAMENTO
# ============================================================

pipeline_xgb = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),
        (
            "modelo",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE
            )
        )
    ]
)


grade_parametros_xgb = {

    "modelo__n_estimators": [
        100,
        200
    ],

    "modelo__max_depth": [
        3,
        5
    ],

    "modelo__learning_rate": [
        0.05,
        0.10
    ]
}


grid_xgb = GridSearchCV(

    estimator=pipeline_xgb,

    param_grid=grade_parametros_xgb,

    scoring="average_precision",

    cv=validacao,

    n_jobs=-1,

    verbose=1
)


grid_xgb.fit(
    X_train,
    y_train
)


modelo_xgb = (
    grid_xgb.best_estimator_
)


print("\n========================================")
print("MELHOR XGBOOST")
print("========================================")

print(
    grid_xgb.best_params_
)

print(
    "PR-AUC CV:",
    grid_xgb.best_score_
)

#%%
# ============================================================
# 19. XGBOOST - BALANCEADO
# ============================================================

n_negativos = (
    y_train == 0
).sum()

n_positivos = (
    y_train == 1
).sum()


scale_pos_weight = (
    n_negativos /
    n_positivos
)


print("\n========================================")
print("PESO DA CLASSE POSITIVA - XGBOOST")
print("========================================")

print(
    f"scale_pos_weight = {scale_pos_weight:.4f}"
)


pipeline_xgb_balanceado = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessamento
        ),
        (
            "modelo",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE
            )
        )
    ]
)


grid_xgb_balanceado = GridSearchCV(

    estimator=pipeline_xgb_balanceado,

    param_grid=grade_parametros_xgb,

    scoring="average_precision",

    cv=validacao,

    n_jobs=-1,

    verbose=1
)


grid_xgb_balanceado.fit(
    X_train,
    y_train
)


modelo_xgb_balanceado = (
    grid_xgb_balanceado.best_estimator_
)


print("\n========================================")
print("MELHOR XGBOOST BALANCEADO")
print("========================================")

print(
    grid_xgb_balanceado.best_params_
)

print(
    "PR-AUC CV:",
    grid_xgb_balanceado.best_score_
)

#%%
# ============================================================
# 20. LISTA DOS SEIS MODELOS
# ============================================================

modelos = {

    "M1 - Regressão Logística":
        modelo_rl,

    "M2 - Regressão Logística Balanceada":
        modelo_rl_balanceado,

    "M3 - Random Forest":
        modelo_rf,

    "M4 - Random Forest Balanceado":
        modelo_rf_balanceado,
    
    "M5 - XGBoost":
        modelo_xgb,

    "M6 - XGBoost Balanceado":
        modelo_xgb_balanceado
}

#%%
# ============================================================
# 21. FUNÇÃO PARA CALCULAR MÉTRICAS
# ============================================================

resultados = []

dados_curvas = {}


for nome_modelo, modelo in modelos.items():

    previsao = modelo.predict(
        X_test
    )

    probabilidade = (
        modelo.predict_proba(
            X_test
        )[:, 1]
    )


    accuracy = accuracy_score(
        y_test,
        previsao
    )

    precision = precision_score(
        y_test,
        previsao,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        previsao,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        previsao,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilidade
    )

    pr_auc = average_precision_score(
        y_test,
        probabilidade
    )


    matriz = confusion_matrix(
        y_test,
        previsao
    )

    tn, fp, fn, tp = (
        matriz.ravel()
    )


    resultados.append({

        "Modelo":
            nome_modelo,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })


    fpr, tpr, _ = roc_curve(
        y_test,
        probabilidade
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            probabilidade
        )
    )


    dados_curvas[nome_modelo] = {

        "fpr":
            fpr,

        "tpr":
            tpr,

        "precision":
            precision_curve,

        "recall":
            recall_curve
    }


    print(
        "\n========================================"
    )

    print(nome_modelo)

    print(
        "========================================"
    )

    print(
        classification_report(
            y_test,
            previsao,
            digits=4
        )
    )

#%%
# ============================================================
# 22. DATAFRAME FINAL DE RESULTADOS
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)


print("\n========================================")
print("RESULTADOS FINAIS")
print("========================================")

print(
    df_resultados[
        [
            "Modelo",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].round(4)
)

#%%
# ============================================================
# 23. SALVAR RESULTADOS EM EXCEL
# ============================================================

arquivo_excel = os.path.join(
    PASTA_RESULTADOS,
    "Resultados_Modelos_AI4I.xlsx"
)


with pd.ExcelWriter(
    arquivo_excel,
    engine="openpyxl"
) as writer:

    df_resultados.to_excel(
        writer,
        sheet_name="Metricas",
        index=False
    )

    estatisticas.to_excel(
        writer,
        sheet_name="Estatisticas"
    )

    correlacao.to_excel(
        writer,
        sheet_name="Correlacao"
    )

    tabela_tipo.to_excel(
        writer,
        sheet_name="Falhas_Tipo"
    )


print(
    "\nArquivo Excel salvo em:"
)

print(
    arquivo_excel
)

#%%
# ============================================================
# 24. MATRIZES DE CONFUSÃO
# ============================================================

for numero, (
    nome_modelo,
    modelo
) in enumerate(
    modelos.items(),
    start=1
):

    previsao = modelo.predict(
        X_test
    )

    matriz = confusion_matrix(
        y_test,
        previsao
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    imagem = ax.imshow(
        matriz
    )


    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(
                    matriz[i, j]
                ),
                ha="center",
                va="center",
                fontsize=15
            )


    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Sem falha",
            "Falha"
        ]
    )


    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "Sem falha",
            "Falha"
        ]
    )


    ax.set_xlabel(
        "Classe prevista"
    )

    ax.set_ylabel(
        "Classe real"
    )


    ax.set_title(
        f"Matriz de confusão\n{nome_modelo}"
    )


    plt.colorbar(
        imagem,
        ax=ax
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            PASTA_RESULTADOS,
            f"06_{numero}_matriz_confusao.png"
        ),
        bbox_inches="tight"
    )


    plt.show()

#%%
# ============================================================
# 25. CURVAS ROC
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


for nome_modelo, dados in (
    dados_curvas.items()
):

    valor_auc = df_resultados.loc[
        df_resultados["Modelo"]
        == nome_modelo,
        "ROC-AUC"
    ].iloc[0]


    ax.plot(
        dados["fpr"],
        dados["tpr"],
        label=(
            f"{nome_modelo} "
            f"(AUC={valor_auc:.3f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


ax.set_xlabel(
    "Taxa de falsos positivos"
)

ax.set_ylabel(
    "Taxa de verdadeiros positivos"
)

ax.set_title(
    "Curvas ROC dos modelos"
)

ax.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "07_curvas_roc.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 26. CURVAS PRECISION-RECALL
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


for nome_modelo, dados in (
    dados_curvas.items()
):

    valor_pr = df_resultados.loc[
        df_resultados["Modelo"]
        == nome_modelo,
        "PR-AUC"
    ].iloc[0]


    ax.plot(
        dados["recall"],
        dados["precision"],
        label=(
            f"{nome_modelo} "
            f"(PR-AUC={valor_pr:.3f})"
        )
    )


ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "Curvas Precision-Recall"
)

ax.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "08_curvas_precision_recall.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 27. COMPARAÇÃO DAS MÉTRICAS
# ============================================================

metricas_grafico = (
    df_resultados
    .set_index("Modelo")
    [
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
    ]
)


metricas_grafico.plot(
    kind="bar",
    figsize=(11, 6)
)


plt.title(
    "Comparação do desempenho dos modelos"
)

plt.ylabel(
    "Valor da métrica"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(
    loc="best"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "09_comparacao_metricas.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 28. COMPARAÇÃO ESPECÍFICA DE RECALL E F1
# ============================================================

df_resultados.set_index(
    "Modelo"
)[
    [
        "Recall",
        "F1"
    ]
].plot(
    kind="bar",
    figsize=(10, 6)
)


plt.title(
    "Comparação de Recall e F1-score"
)

plt.ylabel(
    "Valor"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "10_recall_f1.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 29. IMPORTÂNCIA DAS VARIÁVEIS
# RANDOM FOREST BALANCEADO
# ============================================================

preprocessor_rf = (
    modelo_rf_balanceado
    .named_steps[
        "preprocessamento"
    ]
)


modelo_rf_final = (
    modelo_rf_balanceado
    .named_steps[
        "modelo"
    ]
)


nomes_variaveis = (
    preprocessor_rf
    .get_feature_names_out()
)


importancias = pd.Series(

    modelo_rf_final
    .feature_importances_,

    index=nomes_variaveis

).sort_values(
    ascending=True
)


print("\n========================================")
print("IMPORTÂNCIA DAS VARIÁVEIS")
print("========================================")

print(
    importancias
    .sort_values(
        ascending=False
    )
)


importancias.plot(
    kind="barh",
    figsize=(9, 6)
)


plt.title(
    "Importância das variáveis no Random Forest balanceado"
)

plt.xlabel(
    "Importância"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "11_importancia_variaveis_random_forest.png"
    ),
    bbox_inches="tight"
)

plt.show()

# ============================================================
# XGBOOST BALANCEADO
# ============================================================

preprocessor_xgb = (
    modelo_xgb_balanceado
    .named_steps[
        "preprocessamento"
    ]
)


modelo_xgb_final = (
    modelo_xgb_balanceado
    .named_steps[
        "modelo"
    ]
)


nomes_variaveis_xgb = (
    preprocessor_xgb
    .get_feature_names_out()
)


importancias_xgb = pd.Series(

    modelo_xgb_final
    .feature_importances_,

    index=nomes_variaveis_xgb

).sort_values(
    ascending=True
)


print("\n========================================")
print("IMPORTÂNCIA DAS VARIÁVEIS - XGBOOST")
print("========================================")

print(
    importancias_xgb
    .sort_values(
        ascending=False
    )
)


importancias_xgb.plot(
    kind="barh",
    figsize=(9, 6)
)


plt.title(
    "Importância das variáveis no XGBoost balanceado"
)

plt.xlabel(
    "Importância"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "14_importancia_variaveis_xgboost.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 30. COEFICIENTES DA REGRESSÃO LOGÍSTICA
# ============================================================

preprocessor_rl = (
    modelo_rl_balanceado
    .named_steps[
        "preprocessamento"
    ]
)


modelo_logistico_final = (
    modelo_rl_balanceado
    .named_steps[
        "modelo"
    ]
)


nomes_rl = (
    preprocessor_rl
    .get_feature_names_out()
)


coeficientes = pd.Series(

    modelo_logistico_final
    .coef_[0],

    index=nomes_rl

).sort_values()


print("\n========================================")
print("COEFICIENTES DA REGRESSÃO LOGÍSTICA")
print("========================================")

print(
    coeficientes
)


coeficientes.plot(
    kind="barh",
    figsize=(9, 6)
)


plt.axvline(
    0
)


plt.title(
    "Coeficientes da Regressão Logística balanceada"
)

plt.xlabel(
    "Coeficiente"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        "12_coeficientes_regressao_logistica.png"
    ),
    bbox_inches="tight"
)

plt.show()

#%%
# ============================================================
# 31. GRID SEARCH - SALVAR RESULTADOS
# ============================================================

resultados_grid = pd.DataFrame(
    {
        "Modelo": [
            "Random Forest",
            "Random Forest Balanceado",
            "XGBoost",
            "XGBoost Balanceado"
        ],

        "Melhores parâmetros": [
            str(
                grid_rf.best_params_
            ),

            str(
                grid_rf_balanceado.best_params_
            ),

            str(
                grid_xgb.best_params_
            ),

            str(
                grid_xgb_balanceado.best_params_
            )
        ],

        "PR-AUC validação cruzada": [
            grid_rf.best_score_,
            grid_rf_balanceado.best_score_,
            grid_xgb.best_score_,
            grid_xgb_balanceado.best_score_
        ]
    }
)


resultados_grid.to_excel(
    os.path.join(
        PASTA_RESULTADOS,
        "13_resultados_grid_search.xlsx"
    ),
    index=False
)


print("\n========================================")
print("GRID SEARCH")
print("========================================")

print(
    resultados_grid
)

#%%
# ============================================================
# 32. RESUMO FINAL
# ============================================================

print("\n")
print("=" * 70)

print(
    "PROCESSAMENTO CONCLUÍDO"
)

print("=" * 70)

print(
    "\nTodos os gráficos e tabelas foram salvos em:"
)

print(
    PASTA_RESULTADOS
)

print(
    "\nTabela principal de métricas:"
)

print(
    df_resultados[
        [
            "Modelo",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].round(4)
)