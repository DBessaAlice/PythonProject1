# -------------------------------------------------------------
# Instalação de pacotes necessários (caso não tenha no ambiente)
# -------------------------------------------------------------
# !pip install catboost xlsxwriter scikit-optimize xgboost lightgbm openpyxl

# -------------------------------------------------------------
# Importações de bibliotecas
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import time
import warnings

# Ferramentas de validação cruzada e métricas
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classificadores clássicos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Classificadores ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Classificadores avançados (boosting modernos)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Otimização Bayesiana de hiperparâmetros
from skopt import BayesSearchCV

# Suprimir avisos para não poluir a saída
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. Carregar o dataset (GLOBAL)
# -------------------------------------------------------------
caminho_entrada = 'shallue_all_global.csv'   # Caminho do arquivo de dados
dados = pd.read_csv(caminho_entrada, sep=",")

# Última coluna é o rótulo (classe a prever)
y = dados.iloc[:, -1]

# Demais colunas são os preditores (atributos de entrada)
X = dados.iloc[:, :-1]

# Normalizando os dados (0 a 1 por linha)
# Isso evita que atributos em escalas diferentes influenciem o modelo de forma desbalanceada
X_norm = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1).values

# Convertendo os rótulos para binário (0/1)
lb = LabelBinarizer()
y_bin = lb.fit_transform(y).reshape(-1)

# -------------------------------------------------------------
# 2. Função para rodar experimentos com validação cruzada
# -------------------------------------------------------------
def rodar_experimento(nome_modelo, modelo, parametros, X, y):
    """
    Executa um experimento com:
      - Validação cruzada aninhada (nested cross-validation).
      - Busca Bayesiana de hiperparâmetros (BayesSearchCV).
      - Cálculo de métricas de avaliação.
      - Exportação dos resultados para Excel.
    """

    resultados = []  # Lista para armazenar os resultados

    # Validação cruzada externa (mede a performance do modelo final)
    # - 5 folds (divisões)
    # - 10 repetições para mais robustez
    cv_externa = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

    # Loop da validação cruzada externa
    for treino_idx, teste_idx in cv_externa.split(X, y):
        # Separa conjuntos de treino e teste
        X_treino, X_teste = X[treino_idx], X[teste_idx]
        y_treino, y_teste = y[treino_idx], y[teste_idx]

        # Validação cruzada interna (usada apenas para ajuste de hiperparâmetros)
        cv_interna = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        # Busca Bayesiana (otimização de hiperparâmetros)
        busca = BayesSearchCV(
            modelo, parametros,
            scoring='accuracy',   # métrica de avaliação principal
            cv=cv_interna,        # validação interna
            n_iter=10,            # número de tentativas da busca
            refit=True,           # refaz treino com os melhores parâmetros
            random_state=1,
            n_jobs=-1             # usa todos os núcleos disponíveis
        )

        # Ajuste (treinamento + seleção de parâmetros)
        busca.fit(X_treino, y_treino)

        # Avaliação no conjunto de teste (externo)
        y_pred = busca.predict(X_teste)

        # Guarda métricas e melhores parâmetros
        resultados.append({
            'modelo': nome_modelo,
            'melhores_parametros': busca.best_params_,
            'acuracia': accuracy_score(y_teste, y_pred),
            'precisao': precision_score(y_teste, y_pred),
            'revocacao': recall_score(y_teste, y_pred),
            'f1': f1_score(y_teste, y_pred)
        })

    # Exportar resultados para Excel (um arquivo por modelo)
    pd.DataFrame(resultados).to_excel(f"Classicos/Global/{nome_modelo}.xlsx", index=False)

# -------------------------------------------------------------
# 3. Lista de modelos e hiperparâmetros
# -------------------------------------------------------------
# Aqui definimos os algoritmos e o espaço de busca dos hiperparâmetros.
# Exemplo: quantos vizinhos no KNN, profundidade máxima da árvore, taxa de aprendizado em boosting, etc.

modelos_parametros = {
    'Regressao_Logistica': {
        'modelo': LogisticRegression(),
        'parametros': {'C': (1e-6, 1e+6, 'log-uniform'), 'max_iter': (100, 1000)}
    },
    'KNN': {
        'modelo': KNeighborsClassifier(),
        'parametros': {'n_neighbors': (1, 30)}
    },
    'Naive_Bayes': {
        'modelo': GaussianNB(),
        'parametros': {}  # não tem hiperparâmetros relevantes
    },
    'Arvore_Decisao': {
        'modelo': DecisionTreeClassifier(),
        'parametros': {'max_depth': (1, 20), 'criterion': ['gini', 'entropy']}
    },
    'SVM': {
        'modelo': SVC(),
        'parametros': {'C': (1e-6, 1e+6, 'log-uniform'), 'kernel': ['linear', 'rbf']}
    },
    'Random_Forest': {
        'modelo': RandomForestClassifier(),
        'parametros': {'n_estimators': (10, 200), 'max_depth': (1, 20)}
    },
    'Extra_Trees': {
        'modelo': ExtraTreesClassifier(),
        'parametros': {'n_estimators': (10, 200), 'max_depth': (1, 20)}
    },
    'AdaBoost': {
        'modelo': AdaBoostClassifier(),
        'parametros': {'n_estimators': (10, 200), 'learning_rate': (1e-3, 1, 'log-uniform')}
    },
    'Gradient_Boosting': {
        'modelo': GradientBoostingClassifier(),
        'parametros': {'n_estimators': (10, 200), 'learning_rate': (1e-3, 1, 'log-uniform')}
    },
    'XGBoost': {
        'modelo': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'parametros': {'n_estimators': (10, 200), 'learning_rate': (1e-3, 1, 'log-uniform')}
    },
    'LightGBM': {
        'modelo': LGBMClassifier(),
        'parametros': {'n_estimators': (10, 200), 'learning_rate': (1e-3, 1, 'log-uniform')}
    },
    'CatBoost': {
        'modelo': CatBoostClassifier(verbose=0),
        'parametros': {'depth': (1, 10), 'learning_rate': (1e-3, 1, 'log-uniform')}
    }
}

# -------------------------------------------------------------
# 4. Executar os experimentos
# -------------------------------------------------------------
# Para cada modelo definido acima:
#   - roda o experimento (nested CV + BayesSearchCV)
#   - salva os resultados em Excel
for nome, mp in modelos_parametros.items():
    print(f"Rodando experimento com: {nome}")
    rodar_experimento(nome, mp['modelo'], mp['parametros'], X_norm, y_bin)
