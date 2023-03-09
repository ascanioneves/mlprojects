"""
Atividade para trabalhar o pré-processamento dos dados.
Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.
@author: Aydano Machado <aydano.machado@gmail.com>
"""
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')
#data = data.drop('Pregnancies', axis=1) # -> Drop pregnancies from dataset
#data = data.drop('SkinThickness', axis=1) # -> Drop SkinThickness

data = data.drop('BloodPressure', axis=1)
#data.drop_duplicates(inplace=True)
#data = data.drop('Pregnancies', axis=1) # -> Drop pregnancies from dataset
#data = data.drop('SkinThickness', axis=1) # -> Drop SkinThickness
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Remove all rows that have missing data (initial test, just to see the model behavior)
feature_cols = ['Pregnancies', 'Glucose', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']



imputer = SimpleImputer(strategy='mean')
imputer.fit(data)
imputed_data = imputer.transform(data)
df = pd.DataFrame(imputed_data, columns=data.columns)
X = df[feature_cols]
y = df.Outcome

columns_to_normalize = X.columns

scaler = MinMaxScaler()
X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize]) #normalize all columns
print(X)
#selector = SelectKBest(f_classif, k=7)
#X_new = selector.fit_transform(X, y)
#print(selector.get_support())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
a = accuracy_score(y_test, y_pred)


print(a)
