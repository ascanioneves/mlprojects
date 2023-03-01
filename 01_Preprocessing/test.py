
"""
Atividade para trabalhar o pré-processamento dos dados.
Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.
@author: Aydano Machado <aydano.machado@gmail.com>
"""
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Remove all rows that have missing data (initial test, just to see the model behavior)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#data_dropped = data.dropna(axis=0, how='any')
X = data[feature_cols]
y = data.Outcome
#print(y)

#Splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Combining X_test and y_test with the purpose of dropping test set missing data
combined = pd.concat([X_test, y_test], axis=1)

#Dropping missing values of test set
dropped_test = combined.dropna(axis=0, how='any')
X_test = dropped_test[feature_cols]
y_test = dropped_test.Outcome
# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')

#Combining X_train and y_train with the pyrpose of dropping train set missing data
combined = pd.concat([X_train, y_train], axis=1)
dropped_train = combined.dropna(axis=0, how='any')
X_train = dropped_train[feature_cols]
y_train = dropped_train.Outcome
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#realizando previsões com o arquivo de
y_pred = neigh.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Enviando previsões realizadas com o modelo para o servidor
#URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
#DEV_KEY = "EAE"

# json para ser enviado para o servidor
#data = {'dev_key':DEV_KEY,
#       'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
#r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
#pastebin_url = r.text
#print(" - Resposta do servidor:\n", r.text, "\n")
