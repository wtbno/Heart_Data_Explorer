# Heart Disease Explorer

Dashboard para análise exploratória do dataset Heart Disease, disponibilizado pelo UCI Machine Learning Repository. O projeto foi criado para estudar padrões clínicos e demográficos, comparar grupos e testar uma abordagem simples de modelagem preditiva.

## Sobre o projeto

Este projeto serve como uma aplicação prática de Python, análise de dados e visualização interativa. A ideia principal é explorar as variáveis do conjunto de dados, entender melhor o comportamento dos pacientes e apresentar os resultados em um painel acessível.

Os principais pontos abordados são:

- análise descritiva das variáveis
- comparação entre grupos
- visualizações interativas com Plotly e Streamlit
- testes estatísticos exploratórios
- modelagem preditiva com regressão logística

## Fonte dos dados

- Dataset: Heart Disease
- Repositório: UCI Machine Learning Repository
- DOI: 10.24432/C52P4X
- Base utilizada: Cleveland database, com 14 atributos clássicos do problema

## Importante sobre a variável alvo

A variável original `num` foi mantida conforme a documentação oficial da UCI:

- `0`: ausência de doença (`< 50% de estreitamento em vasos principais`)
- `1, 2, 3, 4`: presença de doença (`> 50% de estreitamento`)

A documentação oficial não justifica interpretar `1, 2, 3, 4` como um nível crescente de gravidade clínica. Por isso, para a análise do dashboard, foi criada uma classificação binária clara:

- `Sem doença`
- `Com doença`

Esse projeto é educacional e não deve ser usado para diagnóstico, triagem clínica ou tomada de decisão médica.

## Tecnologias

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- SciPy
- scikit-learn
- UCI ML Repository

## Estrutura do projeto

```text
HealthData-Explorer/
├── app.py                     # dashboard principal em Streamlit
├── prepare_data.py           # carregamento e preparação do dataset
├── requirements.txt          # dependências do projeto
├── README.md                 # documentação do projeto
├── data/
│   └── heart_disease.csv     # base local do dataset
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
├── outputs/
├── tests/
│   └── test_prepare_data.py
└── .venv/
```

## Método e abordagem

### Carregamento dos dados

O script [prepare_data.py](prepare_data.py) busca a base pública da UCI e salva uma cópia local em `data/heart_disease.csv`.

### Limpeza e organização

- renomeação de colunas para nomes mais legíveis
- conversão de variáveis categóricas para labels semânticas
- preservação da variável original `num` da UCI
- criação de uma variável binária para presença/ausência de doença
- tratamento de valores ausentes no pipeline da modelagem
- manutenção dos registros sem remoção arbitrária

### Modelagem

O modelo preditivo usa regressão logística com:

- divisão entre treino e teste com `stratify`
- imputação de valores ausentes
- codificação de variáveis categóricas
- padronização de variáveis numéricas
- avaliação por acurácia, precisão, recall, F1 e matriz de confusão
- comparação com um baseline simples

## Como executar

### 1) Criar o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3) Carregar o dataset

```bash
python prepare_data.py
```

### 4) Rodar o dashboard

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## Funcionalidades

- filtros interativos por idade, sexo e tipo de dor
- KPIs com contagem e proporção
- histogramas e boxplots
- comparação entre grupos
- matriz de correlação
- testes estatísticos exploratórios
- modelo preditivo educacional
- exportação dos dados filtrados para CSV

## Limitações

- o conjunto de dados é um benchmark clássico e não representa a população geral
- os dados vêm de uma base específica e podem ter viés de amostragem
- correlações e diferenças observadas não implicam causalidade
- o modelo é didático e não substitui avaliação clínica profissional
- valores ausentes em `ca` e `thal` exigem atenção na análise e na interpretação

## Observações finais

Este projeto foi desenvolvido para fins didáticos, de estudo e apresentação em portfólio. O objetivo é demonstrar uso prático de Python, ciência de dados e visualização analítica, sem apresentar conclusões clínicas definitivas ou diagnósticos médicos.
