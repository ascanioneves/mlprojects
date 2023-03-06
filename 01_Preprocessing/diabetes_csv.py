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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

#data = data.drop('Pregnancies', axis=1) # -> Drop pregnancies from dataset
data = data.drop('SkinThickness', axis=1) # -> Drop SkinThickness
#data = data.drop('Insulin')
# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Remove all rows that have missing data (initial test, just to see the model behavior)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',  
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#X = data[feature_cols]
#y = data.Outcome

# Find the columns with missing values
#cols_with_missing = [col for col in data.columns if data[col].isnull().any()]

# Compute the correlation matrix
#corr_matrix = data.corr()

# Compute the correlation between the columns with missing values and the other columns
#corr_with_missing = corr_matrix.loc[cols_with_missing, :]

# Identify the columns with the highest correlation with the columns with missing values
#highest_corr_cols = corr_with_missing.abs().idxmax(axis=1)

# Impute the missing values using the mean of the columns with the highest correlation
#for col in cols_with_missing:
#    data[col].fillna(data[highest_corr_cols[col]].median(), inplace=True)

imputer = SimpleImputer(strategy='mean')
imputer.fit(data)
imputed_data = imputer.transform(data)
df = pd.DataFrame(imputed_data, columns=data.columns)
X = df[feature_cols]
y = df.Outcome

columns_to_normalize = X.columns

scaler = MinMaxScaler()
X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize]) #normalize all columns

#selector = SelectKBest(f_classif, k=7)
#X_new = selector.fit_transform(X, y)


#pca = PCA(n_components=2)
#X = pca.fit_transform(X)

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
#scaler = MinMaxScaler()
#data_app[columns_to_normalize] = scaler.fit_transform(data_app[columns_to_normalize])
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

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
