#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('abalone_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
data = data.drop('length', axis=1) #high correlation with diameter (0.99)

feature_cols = ['sex','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']

#char -> numbers
data['sex'] = data['sex'].map({'M': 0, 'F': 1, 'I': 2})

X = data[feature_cols]
y = data.type

#print(X, y)

#Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('abalone_app.csv')
data_app['sex'] = data_app['sex'].map({'M': 0, 'F': 1, 'I': 2})
data_app = data_app[feature_cols]
y_pred = clf.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "EAE"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")