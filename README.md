# 🫀 Heart Disease Explorer

Dashboard exploratório para análise do dataset Heart Disease, disponibilizado pelo UCI Machine Learning Repository. O projeto foi estruturado como um portfólio profissional para demonstrar boas práticas em análise exploratória de dados, visualização, estatística e modelagem preditiva.

## Objetivo

Explorar padrões clínicos e epidemiológicos associados à presença de doença cardíaca em um conjunto de dados público e educacional. O foco está em:

- análise descritiva das variáveis clínicas
- comparação entre grupos
- visualização interativa com Plotly e Streamlit
- validação estatística em nível exploratório
- modelagem preditiva com regressão logística

## Fonte dos dados

- Dataset: Heart Disease
- Repositório: UCI Machine Learning Repository
- DOI: 10.24432/C52P4X
- referência: Cleveland database, subset de 14 atributos utilizados em experimentos clássicos do problema

## Importante sobre a variável alvo

A variável original `num` representa o diagnóstico angiográfico da doença cardíaca e segue a documentação oficial do UCI:

- `0`: ausência de doença (`< 50% de estreitamento em vasos principais`)
- `1, 2, 3, 4`: presença de doença (`> 50% de estreitamento`)

A documentação não justifica interpretar `1, 2, 3, 4` como “leve, moderada, grave e muito grave” em sentido clínico obrigatório. Por isso, o dashboard utiliza uma classificação binária claramente documentada para a análise de grupos e para o modelo:

- `Sem doença`
- `Com doença`

Este projeto é estritamente educacional e não deve ser usado para diagnosticar, classificar, triagem clínica ou tomada de decisão médica.

## Tecnologias

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- SciPy
- scikit-learn
- UCI ML Repo

## Estrutura da pasta

```text
HealthData-Explorer/
├── app.py                     # dashboard Streamlit
├── prepare_data.py           # carregamento e exportação do dataset
├── requirements.txt          # dependências do projeto
├── README.md                 # documentação do projeto
├── data/
│   └── heart_disease.csv     # dataset baixado da UCI
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
├── outputs/
├── tests/
│   └── test_prepare_data.py
└── .venv/
```

## Métodos e metodologia

### Carregamento dos dados

O script [prepare_data.py](prepare_data.py) acessa a base publicamente disponível pela UCI e salva a versão local em `data/heart_disease.csv`.

### Limpeza e tratamento

- renomeação das colunas para textos mais legíveis
- conversão de códigos categóricos para labels semânticas
- preservação da variável original da UCI (`num`)
- criação de uma variável binária documental para presença/ausência de doença
- imputação de valores ausentes no pipeline de modelagem
- ausência de remoção arbitrária de registros

### Modelagem

O modelo preditivo utiliza regressão logística com:

- divisão treino/teste com `stratify`
- imputação de valores ausentes
- codificação de variáveis categóricas
- padronização de variáveis numéricas
- avaliação com acurácia, precisão, recall, F1 e matriz de confusão
- comparação com um baseline simples

## Como executar

### 1) Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

### 3) Carregar o dataset

```bash
python prepare_data.py
```

### 4) Executar o dashboard

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## Funcionalidades

- filtros interativos por idade, sexo e tipo de dor
- KPIs com contagem e proporção
- distribuições numéricas e boxplots
- comparações entre grupos
- matrizes de correlação
- testes estatísticos exploratórios
- modelo preditivo educacional
- exportação dos dados filtrados para CSV

## Limitações

- o conjunto é um dataset clássico de referência e não representa a população geral
- dados de um hospital/centro específico, com possível viés de amostragem
- não se deve inferir causalidade a partir de associações observadas
- o modelo é didático e não substitui avaliação clínica profissional
- ausência de valores em `ca` e `thal` exige atenção metodológica e validação adicional

## Sugestões de imagens

Para enriquecer o portfólio, recomenda-se incluir screenshots do dashboard em:

- visão geral do painel
- filtros aplicados em idade e sexo
- distribuição por variável clínica
- matriz de correlação
- seção de modelos preditivos

## Observações finais

Este projeto foi desenvolvido para fins didáticos, de estudo e apresentação em portfólio profissional. A intenção é mostrar domínio em Python, ciência de dados e visualização analítica, sem apresentar conclusões clínicas definitivas ou diagnósticos médicos.
