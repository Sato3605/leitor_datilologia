import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle

# Carregar os dados
dados = pd.read_csv('dados_libras.csv')
X = dados.drop('letra', axis=1)
y = dados['letra']
print(f"Amostras carregadas: {dados.shape[0]}")

# Definição dos parâmetros
parametros = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

modelo_base = RandomForestClassifier(random_state=42)
busca = GridSearchCV(modelo_base, parametros, cv=5, verbose=1, n_jobs=1)

print("\nBuscando melhores hiperparâmetros...")
busca.fit(X, y)
melhores = busca.best_params_
print(f"Melhor configuração: {melhores}")

print("\nExecutando validação cruzada...")
modelo_validado = RandomForestClassifier(**melhores, random_state=42)
avaliacoes = cross_val_score(modelo_validado, X, y, cv=10, n_jobs=1)

print("Validação finalizada.")
print(f"Acurácias por teste: {avaliacoes}")
print(f"Média: {np.mean(avaliacoes)*100:.2f}%")
print(f"Desvio: {np.std(avaliacoes)*100:.2f}%")

print("\nTreinando modelo final...")
modelo_final = RandomForestClassifier(**melhores, random_state=42)
modelo_final.fit(X, y)

with open('modelo_libras.pkl', 'wb') as f:
    pickle.dump(modelo_final, f)

print("Modelo salvo com sucesso.")
