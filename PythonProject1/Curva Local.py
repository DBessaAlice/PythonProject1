# ==========================================
# Importações
# ==========================================
import os
import pandas as pd
import numpy as np
from threading import Thread

# Scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Otimização Bayesiana
from skopt import BayesSearchCV

# Modelos de exemplo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================================
# 1. Carregar dataset (LOCAL)
# ==========================================
caminho_entrada_local = 'shallue_all_local.csv'
dados_local = pd.read_csv(caminho_entrada_local, sep=",")

# Separar preditores (X) e rótulos (y)
y_local = dados_local.iloc[:, -1]
X_local = dados_local.iloc[:, :-1]

# Normalização por coluna (cada feature entre 0 e 1)
X_local_norm = X_local.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0).values

# Binarização dos rótulos
lb = LabelBinarizer()
y_local_bin = lb.fit_transform(y_local).reshape(-1)

# ==========================================
# 2. Função para rodar experimentos
# ==========================================
def rodar_experimento_local(nome_modelo, modelo, parametros, X, y):
    resultados = []
    cv_externa = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)

    for treino_idx, teste_idx in cv_externa.split(X, y):
        X_treino, X_teste = X[treino_idx], X[teste_idx]
        y_treino, y_teste = y[treino_idx], y[teste_idx]

        # Cross-validation interna
        cv_interna = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        # Busca Bayesiana de hiperparâmetros
        busca = BayesSearchCV(
            modelo, parametros,
            scoring='accuracy',
            cv=cv_interna,
            n_iter=10,
            refit=True,
            random_state=1,
            n_jobs=-1
        )

        busca.fit(X_treino, y_treino)
        y_pred = busca.predict(X_teste)

        resultados.append({
            'modelo': nome_modelo,
            'melhores_parametros': busca.best_params_,
            'acuracia': accuracy_score(y_teste, y_pred),
            'precisao': precision_score(y_teste, y_pred, zero_division=0),
            'revocacao': recall_score(y_teste, y_pred, zero_division=0),
            'f1': f1_score(y_teste, y_pred, zero_division=0)
        })

    # Criar pasta de saída
    os.makedirs("Classicos/Local", exist_ok=True)

    # Exportar resultados para Excel
    pd.DataFrame(resultados).to_excel(f"Classicos/Local/{nome_modelo}.xlsx", index=False)

    # Exportar também log em arquivo de texto
    with open("saida.txt", "a") as f:
        f.write(f"{nome_modelo} finalizado\n")

# ==========================================
# 3. Definir modelos e hiperparâmetros
# ==========================================
modelos_parametros = {
    "RandomForest": {
        "modelo": RandomForestClassifier(),
        "parametros": {
            "n_estimators": (50, 300),
            "max_depth": (2, 20)
        }
    },
    "XGBoost": {
        "modelo": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "parametros": {
            "n_estimators": (50, 300),
            "max_depth": (2, 20)
        }
    },
    "LightGBM": {
        "modelo": LGBMClassifier(),
        "parametros": {
            "n_estimators": (50, 300),
            "max_depth": (2, 20)
        }
    }
}

# ==========================================
# 4. Rodar experimentos em paralelo (threads)
# ==========================================
threads = []
for nome, mp in modelos_parametros.items():
    exp = Thread(target=rodar_experimento_local, args=(nome, mp['modelo'], mp['parametros'], X_local_norm, y_local_bin))
    exp.start()
    threads.append(exp)

# Esperar todas as threads terminarem
for t in threads:
    t.join()

# Backup: também rodar de forma sequencial
for nome, mp in modelos_parametros.items():
    rodar_experimento_local(nome, mp['modelo'], mp['parametros'], X_local_norm, y_local_bin)