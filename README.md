pip install pandas matplotlib seaborn scikit-learn

# Análise e Predição de Sobrevivência no Titanic

Este relatório apresenta uma análise do dataset do Titanic, baseado no desafio do Kaggle[](https://www.kaggle.com/competitions/titanic/overview). O notebook IPython anexado (`Main.ipynb`) foi utilizado para processar os dados, treinar modelos de machine learning e gerar predições.

## Tema do Projeto

O desafio proposto envolve a criação de um modelo preditivo para determinar quais características dos passageiros do Titanic estão associadas a uma maior probabilidade de sobrevivência. O objetivo é identificar padrões nos dados, como nome, idade, sexo e classe socioeconômica, para prever quais grupos de pessoas tinham uma probabilidade maior de sobreviver. O desafio destaca a necessidade de desenvolver um modelo capaz de discernir os fatores que influenciaram as chances de sobrevivência dos passageiros.

O dataset inclui dados de treinamento (`Train.csv`) e teste (`Test.csv`), com variáveis como PassengerId, Survived (alvo), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin e Embarked. O foco é prever a sobrevivência (0 = Não sobreviveu, 1 = Sobreviveu) para os passageiros do conjunto de teste.

## Preparação e Tratamento dos Dados

Os dados foram carregados usando Pandas e processados para lidar com valores ausentes, criar novas features e discretizar variáveis contínuas. Abaixo, descrevemos cada campo original e como foi tratado:

- **PassengerId**: Identificador único do passageiro. Não utilizado para modelagem; removido após o carregamento.
- **Survived**: Variável alvo (0 ou 1). Mantida no conjunto de treinamento para treinamento e validação; não presente no teste.
- **Pclass**: Classe socioeconômica (1 = 1ª classe, 2 = 2ª classe, 3 = 3ª classe). Mantida como está, pois já é categórica/ordinal.
- **Name**: Nome do passageiro. Utilizado para extrair o título (e.g., Mr, Miss, Mrs), que foi mapeado para valores numéricos: Mr=1, Miss=2, Mrs=3, Master=4, Prestige=5 (incluindo títulos raros como Lady, Dr, etc.). O campo original foi removido após extração.
- **Sex**: Sexo (male/female). Mapeado para binário: female=0, male=1.
- **Age**: Idade. Valores ausentes preenchidos com a mediana por sexo. Discretizada em 6 bins usando KBinsDiscretizer (estratégia quantile): {0: '[0.42, 19.00)', 1: '[19.00, 25.00)', 2: '[25.00, 29.00)', 3: '[29.00, 31.00)', 4: '[31.00, 40.50)', 5: '[40.50, 80.00]'}.
- **SibSp**: Número de irmãos/cônjuges a bordo. Utilizado para criar FamSize (SibSp + Parch + 1); campo original removido.
- **Parch**: Número de pais/filhos a bordo. Utilizado para criar FamSize; campo original removido.
- **Ticket**: Número do bilhete. Removido, pois não adiciona valor preditivo direto.
- **Fare**: Tarifa paga. Valores ausentes preenchidos com a mediana do treino. Discretizada em 6 bins usando KBinsDiscretizer (estratégia quantile): {0: '[0.00, 7.78)', 1: '[7.78, 8.66)', 2: '[8.66, 14.45)', 3: '[14.45, 26.00)', 4: '[26.00, 52.37)', 5: '[52.37, 512.33]'}.
- **Cabin**: Cabine. Transformada em HasCabin (1 se presente, 0 se ausente); campo original removido.
- **Embarked**: Porto de embarque (S, C, Q). Valores ausentes preenchidos com 'S'; mapeado para: S=0, C=1, Q=2.

Novas features criadas:
- **FamSize**: Tamanho da família (SibSp + Parch + 1).
- **HasCabin**: Indicador de presença de cabine (0 ou 1).
- **Title**: Título extraído do nome (mapeado para 1-5).
- **IsAlone**: Indicador de viagem sozinho (1 se FamSize == 1, 0 caso contrário).

Valores ausentes identificados:
- Treino: Age (177), Cabin (687), Embarked (2).
- Teste: Age (86), Fare (1), Cabin (327).

Após tratamento, os datasets foram limpos e prontos para modelagem. Uma matriz de correlação de Pearson foi plotada para visualizar relações entre features.

## Modelos de Machine Learning Utilizados

Os dados de treinamento foram divididos em XTrain (features) e YTrain (Survived). Quatro modelos de classificação foram treinados e validados usando cross-validation (5 folds) para avaliar desempenho. As métricas incluem accuracy, precision, recall e F1-score. Além disso, matrizes de confusão e relatórios de classificação foram gerados.

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression (LogReg)**
- **Decision Tree**
- **Random Forest**

Treinamento e Validação:
- Cross-validation foi aplicada para estimar o desempenho geral.
- Predições foram geradas no conjunto de teste para submissão no Kaggle.
- Avaliação no treino: Comparação de métricas para selecionar o melhor modelo.

## Resultados Obtidos

- **Métricas Detalhadas**:
  - **KNN**:
    - Accuracy: 0.803
    - Precision: 0.733
    - Recall: 0.75
    - F1-Score: 0.741
    - AUC Score: 0.792
    - AP Score: 0.644
  - **Logistic Regression**:
    - Accuracy: 0.798
    - Precision: 0.71
    - Recall: 0.786
    - F1-Score: 0.746
    - AUC Score: 0.796
    - AP Score: 0.638
  - **Decision Tree**:
    - Accuracy: 0.812
    - Precision: 0.776
    - Recall: 0.702
    - F1-Score: 0.738
    - AUC Score: 0.79
    - AP Score: 0.657
  - **Random Forest**:
    - Accuracy: 0.83
    - Precision: 0.78
    - Recall: 0.762
    - F1-Score: 0.771
    - AUC Score: 0.816
    - AP Score: 0.684

Matrizes de confusão mostraram que os modelos erram mais em falsos negativos (prever não sobrevivência quando sobreviveu), possivelmente devido ao desbalanceamento de classes.

O Random Forest apresentou o melhor desempenho geral, sugerindo que ensembles são eficazes para este problema.

Arquivos de submissão gerados:
- Submission_KNN.csv
- Submission_LogReg.csv
- Submission_DecisionTree.csv
- Submission_RandomForest.csv
