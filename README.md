# Wine Quality Classification
### POSTECH – DTAT – Tech Challenge Fase 2

Classificação binária da qualidade de vinhos a partir de variáveis físico-químicas, usando Machine Learning com Python e scikit-learn.

## Objetivo

Prever se um vinho é de Alta Qualidade (nota ≥ 7) ou Baixa/Média Qualidade (nota < 7), com base em atributos mensuráveis durante a produção, como acidez, teor alcoólico e sulfatos.

## Estrutura do Projeto

```
wine-quality-classification/
│
├── data/
│   └── WineQT.csv
│
├── notebooks/
│   └── wine_quality_analysis.ipynb   # Análise completa
│
├── src/
│   ├── preprocess.py             # Carregamento, limpeza e feature engineering
│
├── results/                      # Gráficos e métricas gerados automaticamente
├── requirements.txt
└── README.md
```


## Dataset

[Wine Quality Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

### Como rodar o pipeline completo (scripts)
```bash
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

## GRUPO 102
Amanda Cristine da Silva Gomes Queiroz

João Paulo Viana Melo

Kethellen Santana Da Silva

Gabrielle Tainá Inácio Oliveira

Rafaela Uchôas
