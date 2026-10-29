import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Heart Disease Explorer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_COLS = [
    "Idade",
    "Pressão arterial em repouso",
    "Colesterol",
    "Frequência cardíaca máxima",
    "Depressão de ST",
    "Vasos principais",
]

CATEGORICAL_COLS = [
    "Sexo",
    "Tipo de dor no peito",
    "Glicemia de jejum > 120",
    "ECG em repouso",
    "Angina induzida por exercício",
    "Inclinação do segmento ST",
    "Talassemia",
]


@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease.csv")

    df = df.rename(
        columns={
            "age": "Idade",
            "sex": "Sexo",
            "cp": "Tipo de dor no peito",
            "trestbps": "Pressão arterial em repouso",
            "chol": "Colesterol",
            "fbs": "Glicemia de jejum > 120",
            "restecg": "ECG em repouso",
            "thalach": "Frequência cardíaca máxima",
            "exang": "Angina induzida por exercício",
            "oldpeak": "Depressão de ST",
            "slope": "Inclinação do segmento ST",
            "ca": "Vasos principais",
            "thal": "Talassemia",
            "num": "Diagnóstico (UCI)",
        }
    )

    df["Sexo"] = df["Sexo"].map({0: "Feminino", 1: "Masculino"})
    df["Tipo de dor no peito"] = df["Tipo de dor no peito"].map(
        {
            1: "Angina típica",
            2: "Angina atípica",
            3: "Dor não anginosa",
            4: "Assintomático",
        }
    )
    df["Glicemia de jejum > 120"] = df["Glicemia de jejum > 120"].map({0: "Não", 1: "Sim"})
    df["ECG em repouso"] = df["ECG em repouso"].map(
        {
            0: "Normal",
            1: "Anormalidade ST-T",
            2: "Hipertrofia ventricular esquerda",
        }
    )
    df["Angina induzida por exercício"] = df["Angina induzida por exercício"].map({0: "Não", 1: "Sim"})
    df["Inclinação do segmento ST"] = df["Inclinação do segmento ST"].map({1: "Ascendente", 2: "Plana", 3: "Descendente"})
    df["Talassemia"] = df["Talassemia"].map({3: "Normal", 6: "Defeito fixo", 7: "Defeito reversível"})
    df["Diagnóstico"] = np.where(df["Diagnóstico (UCI)"] > 0, "Com doença", "Sem doença")
    df["Grupo diagnóstico"] = df["Diagnóstico"].copy()

    for col in [
        "Sexo",
        "Tipo de dor no peito",
        "Glicemia de jejum > 120",
        "ECG em repouso",
        "Angina induzida por exercício",
        "Inclinação do segmento ST",
        "Talassemia",
        "Diagnóstico",
        "Grupo diagnóstico",
    ]:
        df[col] = df[col].astype("category")

    df["Diagnóstico (UCI)"] = pd.to_numeric(df["Diagnóstico (UCI)"], errors="coerce")
    df["Vasos principais"] = pd.to_numeric(df["Vasos principais"], errors="coerce")

    return df


df = load_data()

st.title("🫀 Heart Disease Explorer")
st.markdown(
    """
    Dashboard de **análise exploratória** do dataset *Heart Disease*,
    disponibilizado pelo UCI Machine Learning Repository.

    **Aviso importante:** a variável original `num` da UCI não representa severidade clínica em ordem crescente.
    Para este dashboard, a análise usa uma classificação binária documentada: presença ou ausência de doença.
    """
)

st.divider()

# ------------------------------
# Filtros
# ------------------------------
st.sidebar.header("🔎 Filtros")

idade_min = int(df["Idade"].min())
idade_max = int(df["Idade"].max())

faixa_idade = st.sidebar.slider(
    "Faixa de idade",
    min_value=idade_min,
    max_value=idade_max,
    value=(idade_min, idade_max),
)

sexo = st.sidebar.multiselect(
    "Sexo",
    options=df["Sexo"].dropna().unique().tolist(),
    default=df["Sexo"].dropna().unique().tolist(),
)

grupo_diagnostico = st.sidebar.multiselect(
    "Grupo diagnóstico",
    options=df["Grupo diagnóstico"].dropna().unique().tolist(),
    default=df["Grupo diagnóstico"].dropna().unique().tolist(),
)

tipo_dor = st.sidebar.multiselect(
    "Tipo de dor no peito",
    options=df["Tipo de dor no peito"].dropna().unique().tolist(),
    default=df["Tipo de dor no peito"].dropna().unique().tolist(),
)

df_filtrado = df[
    df["Idade"].between(faixa_idade[0], faixa_idade[1])
    & df["Sexo"].isin(sexo)
    & df["Grupo diagnóstico"].isin(grupo_diagnostico)
    & df["Tipo de dor no peito"].isin(tipo_dor)
]

st.sidebar.divider()
st.sidebar.metric("Registros filtrados", len(df_filtrado))

if df_filtrado.empty:
    st.warning("Nenhum registro corresponde aos filtros selecionados.")
    st.stop()

# ------------------------------
# KPIs
# ------------------------------
col1, col2, col3, col4 = st.columns(4)
com_doenca = (df_filtrado["Grupo diagnóstico"] == "Com doença").mean()

col1.metric("Registros", len(df_filtrado))
col2.metric("Idade média", f"{df_filtrado['Idade'].mean():.1f} anos")
col3.metric("Colesterol médio", f"{df_filtrado['Colesterol'].mean():.1f} mg/dl")
col4.metric("Com doença cardíaca", f"{com_doenca * 100:.1f}%")

st.divider()

(
    tab_dados,
    tab_qualidade,
    tab_distribuicoes,
    tab_comparacoes,
    tab_correlacoes,
    tab_testes,
    tab_modelo,
    tab_metodologia,
) = st.tabs(
    [
        "📋 Dados",
        "🧪 Qualidade dos dados",
        "📊 Distribuições",
        "⚖️ Comparações",
        "🔗 Correlações",
        "📈 Testes estatísticos",
        "🤖 Modelo preditivo",
        "📚 Metodologia",
    ]
)

# ------------------------------
# Dados
# ------------------------------
with tab_dados:
    st.subheader("Amostra dos dados filtrados")
    colunas_tabela = st.multiselect(
        "Selecione as colunas para visualizar",
        options=df_filtrado.columns.tolist(),
        default=[
            "Idade",
            "Sexo",
            "Tipo de dor no peito",
            "Colesterol",
            "Pressão arterial em repouso",
            "Frequência cardíaca máxima",
            "Diagnóstico",
        ],
    )
    st.dataframe(df_filtrado[colunas_tabela].head(50), width="stretch", hide_index=True)
    st.info(
        "📌 O que este gráfico mostra: a tabela apresenta uma amostra dos dados após os filtros. "
        "Cada linha representa um paciente e cada coluna representa uma característica clínica ou demográfica. "
        "É o ponto de partida para entender o perfil da população selecionada."
    )
    st.caption(
        "A tabela mostra os primeiros 50 registros após a aplicação dos filtros. "
        "Use a seleção de colunas para focar nas informações mais relevantes."
    )
    csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Baixar dados filtrados em CSV",
        data=csv,
        file_name="heart_disease_filtrado.csv",
        mime="text/csv",
    )

# ------------------------------
# Qualidade dos dados
# ------------------------------
with tab_qualidade:
    st.subheader("Valores ausentes")
    ausentes = df_filtrado.isnull().sum()
    ausentes = ausentes[ausentes > 0].sort_values(ascending=False)

    if ausentes.empty:
        st.success("Nenhum valor ausente no conjunto filtrado.")
    else:
        resumo_ausentes = pd.DataFrame(
            {
                "Valores ausentes": ausentes,
                "Percentual": (ausentes / len(df_filtrado) * 100).round(2),
            }
        )
        st.dataframe(resumo_ausentes, width="stretch")

        fig_ausentes = px.bar(
            resumo_ausentes.reset_index().rename(columns={"index": "Variável"}),
            x="Percentual",
            y="Variável",
            orientation="h",
            text="Valores ausentes",
            title="Percentual de valores ausentes por variável",
        )
        fig_ausentes.update_layout(template="plotly_white")
        st.plotly_chart(fig_ausentes, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: cada barra representa uma variável e o tamanho da barra indica "
            "quanto daquela coluna ficou vazia. Quando uma variável tem muitos dados faltantes, a análise dessa variável "
            "precisa ser interpretada com mais cuidado."
        )

        st.caption(
            "Barras mais longas indicam maior proporção de valores ausentes. "
            "Essa informação ajuda a avaliar a confiabilidade das análises por variável."
        )

    st.subheader("Registros duplicados")
    duplicados = df_filtrado.duplicated().sum()
    st.metric("Linhas duplicadas", duplicados)

# ------------------------------
# Distribuições
# ------------------------------
with tab_distribuicoes:
    st.subheader("Distribuição de variáveis numéricas")
    variavel = st.selectbox("Variável", NUMERIC_COLS)

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            df_filtrado,
            x=variavel,
            color="Grupo diagnóstico",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            title=f"Histograma — {variavel}",
        )
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: o histograma mostra a frequência com que os valores aparecem. "
            "Se a curva estiver mais concentrada em um ponto, isso indica que a maioria dos pacientes tem valores próximos. "
            "Comparar os grupos ajuda a entender se há diferença de comportamento entre quem tem e quem não tem doença."
        )

        st.caption(
            "Mostra como os valores da variável selecionada estão distribuídos. "
            "Barras mais altas indicam maior concentração de registros; compare as "
            "distribuições entre os grupos com e sem doença."
        )

    with col_box:
        fig_box = px.box(
            df_filtrado,
            x="Grupo diagnóstico",
            y=variavel,
            color="Grupo diagnóstico",
            title=f"Boxplot — {variavel}",
        )
        fig_box.update_layout(template="plotly_white")
        st.plotly_chart(fig_box, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: o boxplot resume a distribuição em uma visão compacta. "
            "A caixa mostra onde fica a maior parte dos valores; a linha no meio representa a mediana; "
            "os pontos fora da caixa podem ser valores muito altos ou muito baixos, chamados de outliers."
        )

        st.caption(
            "A linha central representa a mediana e a caixa mostra a variação central "
            "dos dados. Pontos isolados podem indicar valores atípicos."
        )

    st.subheader("Resumo estatístico")
    resumo = df_filtrado.groupby("Grupo diagnóstico", observed=True)[variavel].describe().round(2)
    st.dataframe(resumo, width="stretch")

    st.caption(
        "O resumo apresenta média, desvio padrão, valores mínimos, máximos e quartis "
        "para cada grupo diagnóstico."
    )

# ------------------------------
# Comparações
# ------------------------------
with tab_comparacoes:
    st.subheader("Comparação entre grupos")
    variavel_categorica = st.selectbox(
        "Variável categórica",
        [
            "Sexo",
            "Tipo de dor no peito",
            "Glicemia de jejum > 120",
            "ECG em repouso",
            "Angina induzida por exercício",
            "Talassemia",
        ],
    )

    contagem = (
        df_filtrado.groupby([variavel_categorica, "Grupo diagnóstico"], observed=True)
        .size()
        .reset_index(name="Registros")
    )

    fig_barras = px.bar(
        contagem,
        x=variavel_categorica,
        y="Registros",
        color="Grupo diagnóstico",
        barmode="group",
        title=f"Registros por {variavel_categorica} e grupo diagnóstico",
    )
    fig_barras.update_layout(template="plotly_white")
    st.plotly_chart(fig_barras, width="stretch")

    st.info(
        "📌 O que este gráfico mostra: cada barra representa uma categoria, como sexo ou tipo de dor, "
        "e a altura da barra mostra quantos pacientes pertencem a ela. "
        "Esse tipo de gráfico facilita comparar grupos e perceber se uma categoria aparece mais frequentemente em um perfil de doença."
    )

    st.caption(
        "Compara a quantidade de registros em cada categoria. "
        "Diferenças de altura mostram quais grupos são mais frequentes no recorte filtrado."
    )

    st.subheader("Relação entre idade e frequência cardíaca máxima")
    fig_scatter = px.scatter(
        df_filtrado,
        x="Idade",
        y="Frequência cardíaca máxima",
        color="Grupo diagnóstico",
        hover_data=["Colesterol", "Pressão arterial em repouso", "Depressão de ST"],
        title="Idade versus frequência cardíaca máxima",
        trendline="ols",
    )
    fig_scatter.update_layout(template="plotly_white")
    st.plotly_chart(fig_scatter, width="stretch")

    st.info(
        "📌 O que este gráfico mostra: cada ponto representa um paciente. "
        "O eixo horizontal mostra a idade e o eixo vertical mostra a frequência cardíaca máxima. "
        "Quando os pontos ficam mais altos ou mais baixos em certos trechos, isso sugere uma tendência. "
        "A linha ajuda a visualizar essa direção geral, mas não prova causa e efeito."
    )

    st.caption(
        "Cada ponto representa um registro. Observe se há alguma tendência entre idade "
        "e frequência cardíaca máxima; a linha indica uma relação linear descritiva, "
        "não uma relação de causa e efeito."
    )

# ------------------------------
# Correlações
# ------------------------------
with tab_correlacoes:
    st.subheader("Correlação entre variáveis numéricas")
    st.caption(
        "A matriz considera apenas variáveis numéricas contínuas. "
        "Correlação não implica causalidade."
    )
    variaveis_corr = st.multiselect(
        "Selecione as variáveis",
        options=NUMERIC_COLS,
        default=NUMERIC_COLS,
    )

    if len(variaveis_corr) < 2:
        st.info("Selecione pelo menos duas variáveis numéricas.")
    else:
        corr = df_filtrado[variaveis_corr].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de correlação",
        )
        fig_corr.update_layout(template="plotly_white")
        st.plotly_chart(fig_corr, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: a matriz de correlação indica quão fortemente duas variáveis se movem juntas. "
            "Valores próximos de 1 significam crescimento conjunto; valores próximos de -1 significam comportamento oposto; "
            "valores perto de 0 sugerem pouca associação linear. Esse gráfico ajuda a descobrir padrões, mas não explica por quê."
        )

        st.caption(
            "Valores próximos de 1 indicam que as variáveis tendem a aumentar juntas; "
            "valores próximos de -1 indicam direções opostas. Valores próximos de 0 "
            "sugerem pouca relação linear."
        )

# ------------------------------
# Testes estatísticos
# ------------------------------
with tab_testes:
    st.subheader("Comparação estatística entre grupos")
    st.caption(
        "Compara variáveis numéricas entre os grupos 'Sem doença' e 'Com doença'. "
        "A escolha do teste depende da distribuição observada e da quantidade de dados."
    )
    variavel_teste = st.selectbox("Variável numérica", NUMERIC_COLS)

    grupo_sem = df_filtrado.loc[df_filtrado["Grupo diagnóstico"] == "Sem doença", variavel_teste].dropna()
    grupo_com = df_filtrado.loc[df_filtrado["Grupo diagnóstico"] == "Com doença", variavel_teste].dropna()

    if len(grupo_sem) < 2 or len(grupo_com) < 2:
        st.warning("Selecione filtros que mantenham registros nos dois grupos.")
    else:
        shapiro_sem = stats.shapiro(grupo_sem)
        shapiro_com = stats.shapiro(grupo_com)
        normal_sem = shapiro_sem.pvalue > 0.05
        normal_com = shapiro_com.pvalue > 0.05

        if normal_sem and normal_com:
            estatistica, valor_p = stats.ttest_ind(grupo_com, grupo_sem, equal_var=False)
            teste = "Teste t de Welch"
            pooled_sd = np.sqrt(
                (
                    (len(grupo_sem) - 1) * grupo_sem.var(ddof=1)
                    + (len(grupo_com) - 1) * grupo_com.var(ddof=1)
                )
                / (len(grupo_sem) + len(grupo_com) - 2)
            )
            efeito = (grupo_com.mean() - grupo_sem.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        else:
            estatistica, valor_p = stats.mannwhitneyu(grupo_com, grupo_sem, alternative="two-sided")
            teste = "Teste Mann–Whitney U"
            efeito = np.nan

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Média — Sem doença", f"{grupo_sem.mean():.2f}")
        col2.metric("Média — Com doença", f"{grupo_com.mean():.2f}")
        col3.metric("Estatística", f"{estatistica:.3f}")
        col4.metric("Valor-p", f"{valor_p:.4f}")

        if not np.isnan(efeito):
            st.caption(f"Tamanho de efeito (d de Cohen): {efeito:.3f}")
        else:
            st.caption("Tamanho de efeito não calculado para o teste não paramétrico.")

        if valor_p < 0.05:
            st.success(f"Diferença estatisticamente significativa entre os grupos. Teste aplicado: {teste}.")
        else:
            st.info("Não foi identificada diferença estatisticamente significativa entre os grupos no nível de significância de 5%.")

        fig_teste = px.box(
            df_filtrado,
            x="Grupo diagnóstico",
            y=variavel_teste,
            color="Grupo diagnóstico",
            points="all",
            title=f"{variavel_teste} por grupo diagnóstico",
        )
        fig_teste.update_layout(template="plotly_white")
        st.plotly_chart(fig_teste, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: o boxplot mostra como os valores se distribuem em cada grupo. "
            "O valor-p responde à pergunta: 'a diferença observada parece maior do que o acaso?' "
            "Se o valor-p for pequeno, há evidência de diferença entre os grupos. "
            "Mas isso não quer dizer que um grupo causa o outro; apenas que há associação estatística."
        )

        st.caption(
            "O boxplot compara a distribuição da variável entre os dois grupos. "
            "O valor-p ajuda a avaliar se a diferença observada é estatisticamente "
            "significativa, mas não estabelece causalidade."
        )

# ------------------------------
# Modelo preditivo
# ------------------------------
with tab_modelo:
    st.subheader("Modelo preditivo educacional")
    st.caption(
        "A modelagem usa o conjunto completo de dados, sem depender dos filtros da "
        "exploração visual. Isso evita que a análise exploratória altere silenciosamente "
        "o treinamento e a avaliação do modelo."
    )

    df_modelo = df.copy()
    alvo = "Grupo diagnóstico"
    df_modelo = df_modelo[df_modelo[alvo].notna()]
    y = (df_modelo[alvo] == "Com doença").astype(int)
    X = df_modelo[NUMERIC_COLS + CATEGORICAL_COLS]

    if y.nunique() < 2:
        st.warning(
            "É necessário ter registros dos dois grupos para treinar o modelo. "
            "Ajuste os filtros ou utilize o conjunto completo de dados."
        )
    else:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessamento = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_COLS),
                ("cat", categorical_transformer, CATEGORICAL_COLS),
            ]
        )

        modelo = Pipeline(
            steps=[
                ("preprocessamento", preprocessamento),
                (
                    "classificador",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                ),
            ]
        )

        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_treino, y_treino)
        y_pred_baseline = baseline.predict(X_teste)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Acurácia", f"{accuracy_score(y_teste, y_pred):.1%}")
        col2.metric("Precisão", f"{precision_score(y_teste, y_pred, zero_division=0):.1%}")
        col3.metric("Recall", f"{recall_score(y_teste, y_pred, zero_division=0):.1%}")
        col4.metric("F1-score", f"{f1_score(y_teste, y_pred, zero_division=0):.1%}")
        col5.metric("Baseline", f"{accuracy_score(y_teste, y_pred_baseline):.1%}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring="f1")
        st.metric("F1 em validação cruzada (5-fold)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        st.caption(
            "As métricas do conjunto de teste são reportadas separadamente da exploração do dashboard. "
            "A comparação com o baseline ajuda a avaliar se o modelo captura informação útil além "
            "da classe mais frequente."
        )

        st.subheader("Matriz de confusão")
        matriz = confusion_matrix(y_teste, y_pred)
        fig_matriz = px.imshow(
            matriz,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Previsto", y="Real"),
            x=["Sem doença", "Com doença"],
            y=["Sem doença", "Com doença"],
            title="Matriz de confusão no conjunto de teste",
        )
        fig_matriz.update_layout(template="plotly_white")
        st.plotly_chart(fig_matriz, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: a matriz de confusão compara o que o modelo previu com o que realmente aconteceu. "
            "A diagonal principal mostra acertos e as outras células mostram erros. "
            "Se a maior parte dos casos ficar na diagonal, o modelo está acertando com frequência."
        )

        st.caption(
            "A diagonal principal mostra acertos do modelo. "
            "As células fora da diagonal representam classificações incorretas."
        )

        st.subheader("Relatório de classificação")
        relatorio = classification_report(
            y_teste,
            y_pred,
            target_names=["Sem doença", "Com doença"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(relatorio).T.round(3), width="stretch")

        st.info(
            "📌 O que este quadro mostra: ele resume o desempenho do modelo separadamente para cada classe. "
            "Precisão indica quantos casos previstos como positivos realmente eram positivos; recall mostra quantos casos reais positivos o modelo conseguiu encontrar; "
            "F1 é uma medida combinada que equilibra esses dois aspectos."
        )

        st.caption(
            "O relatório detalha precisão, recall e F1-score para cada classe. "
            "Valores mais próximos de 1 indicam melhor desempenho naquela classe."
        )

        st.subheader("Importância das variáveis")
        categorias_codificadas = modelo.named_steps["preprocessamento"]["cat"].named_steps["encoder"].get_feature_names_out(CATEGORICAL_COLS)
        nomes_variaveis = np.concatenate([NUMERIC_COLS, categorias_codificadas])
        coeficientes = modelo.named_steps["classificador"].coef_[0]

        importancia = (
            pd.DataFrame({"Variável": nomes_variaveis, "Coeficiente": coeficientes})
            .assign(Impacto_absoluto=lambda d: d["Coeficiente"].abs())
            .sort_values("Impacto_absoluto", ascending=False)
            .head(15)
        )

        fig_importancia = px.bar(
            importancia,
            x="Coeficiente",
            y="Variável",
            orientation="h",
            title="Principais variáveis associadas à previsão",
            color="Coeficiente",
            color_continuous_scale="RdBu_r",
        )
        fig_importancia.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_importancia, width="stretch")

        st.info(
            "📌 O que este gráfico mostra: ele destaca as variáveis que mais influenciaram a previsão do modelo. "
            "A altura e a cor indicam o peso daquela variável. Porém, esse resultado precisa ser interpretado com cautela: "
            "os coeficientes dependem do tipo de codificação e do pré-processamento, então não representam uma regra universal de importância clínica."
        )

        st.caption(
            "Coeficientes da regressão logística devem ser interpretados com cuidado, porque "
            "a magnitude e o sinal dependem do pré-processamento, da codificação e da escala "
            "das variáveis. Eles não são uma medida universal de importância causal ou clínica."
        )

# ------------------------------
# Metodologia
# ------------------------------
with tab_metodologia:
    st.subheader("Sobre o projeto")
    st.markdown(
        """
        Este dashboard realiza **análise exploratória** do dataset *Heart Disease*,
        disponibilizado pelo UCI Machine Learning Repository.

        O objetivo é descrever variáveis, identificar padrões, comparar grupos
        e visualizar possíveis associações entre variáveis clínicas.

        A variável original `num` foi preservada e interpretada conforme a documentação
        oficial da UCI: 0 indica ausência de doença e 1-4 indicam presença de doença.
        Não há justificativa para tratar 1, 2, 3 e 4 como escalas clínicas em ordem
        crescente de severidade.
        """
    )

    st.subheader("Como interpretar")
    st.markdown(
        """
        - As visualizações descrevem o conjunto de dados analisado.
        - Associações entre variáveis não indicam relação de causa e efeito.
        - Os filtros visuais não são aplicados ao modelo preditivo, para evitar vazamento de dados.
        - Este projeto é educacional e não deve ser usado para diagnóstico,
          triagem ou decisão clínica.
        """
    )

    st.subheader("Fonte dos dados")
    st.markdown(
        """
        [Heart Disease — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
        """
    )

st.divider()
st.caption(
    "Fonte: UCI Machine Learning Repository — Heart Disease. "
    "Projeto de portfólio para análise exploratória de dados e modelagem preditiva educativa."
)