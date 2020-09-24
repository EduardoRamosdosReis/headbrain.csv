# headbrain.csv
# Cuidado com a saúde

from google.colab import files
upload = files.upload()

import pandas as pd # Trabaljar com data set
import numpy as np # trabalhar com arrays e vetores
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # Faz escalonamento /altera valores categorics para numericos # Preprocessamento
from sklearn.model_selection import train_test_split # faz a divisão de treino e teste de classficação
from sklearn.metrics import confusion_matrix, accuracy_score # demonstra a precisaão e compração entre os algoritmos 
from sklearn.naive_bayes import GaussianNB # aplica o algoritmo Naive Bayes
import seaborn as sns # melhoria dos layouts de gráficos 
import matplotlib.pyplot as plt # plotar dados e gráficos 
from sklearn.svm import SVC # importa o algoritmo Supoort Vetor Machine,  indentifica a fronteira dos dados, hiperplano de separação   

nomeArquivo = "data531.csv" 
dataset = pd.read_csv(nomeArquivo, sep=",") #realiza a leitura do banco de dados, indicando separação de virgulas. 
dataset2 =pd.read_csv("data531.csv")

#visualizar o seu dataset
dataset.head(10)
dataset.shape
dataset.isnull().sum () # somando os valores NaN do data set.
dataset.info()
dataset.fillna(dataset.mean(),inplace=True)# alterando para a media de cada coluna todos os valores nulos, inplace = True substitui todos os valores presentes nos dados
dataset.loc[dataset["age"],"age"].mean() #media da idade serve para todas as colunas

data_set_array = np.array(dataset) # Criando uma variável do tipo array para preencher com o DAtaSet agora do tipo array
#criando a saida com a coluna Target, para que seja nosso y
targety = data_set_array[:,57] # valor de saída
targety = targety.astype("int") #indica o tipo de dados que queremos transformar, nesse caso o array
print(targety)

#dados coletados pelos sensores, CRIA  NO ARRAY COM O DADOS ADVINDOS DOS ITENS RELACIONADOS A SENSORES
data_set_sensores = np.column_stack((
    data_set_array[:,11], # pressão sanguinia
    data_set_array[:,33], # freq max atingida
    data_set_array[:,34], # frea  car repouso
    data_set_array[:,35], # pico de pressão sanguinea exercício
    data_set_array[:,36], # pico de pressão sanguinea exercícios
    data_set_array[:,38]  # pressão sangui em repouso 
))
data_set_sensores

#data set com dados médicos do paciente CRIA UM ARRAY COM O DADOS ADVINDOS DOS DOS MÉDICOS
data_medicos = np.column_stack((
    data_set_array[:,4], #localização da dor
    data_set_array[:,6], # alívio após cansaço 
    data_set_array[:,9], # tipo de dor
    data_set_array[:,39], # angina induzida pelo exercício
    dataset.age,
    dataset.sex,
    dataset.hypertension
))
print (data_medicos)

#concatenar as duas bases de dados
dataset_paciente = np.concatenate((data_set_sensores,data_medicos),axis = 1)
print(dataset_paciente)

dataset_paciente.shape

#separar os valores como dados de train and test
X_train, X_test, y_train,y_test = train_test_split(dataset_paciente,targety, random_state = 223 )
#Criar objeto para a utilziação do SVM função SVC
modelSVM = SVC(kernel= "linear") # transformação que serão utilizados nos dados, os dados devem ser linearmente separdos TESTAR com BINOMINAL POLINOMIAL
modelSVM.fit(X_train,y_train) #criando a tabela ou função SVM, modelo de treinamento 

##Analisando a performance do modelo

previsao = modelSVM.predict(X_test) #realizando o modelo de previsão
#accuracy do modelo de previsão 
accuracy = accuracy_score(y_test,previsao)
percentual = round(accuracy*100)
print("A accurancia do modelo SVM é de {},\n sendo um percentual de {}  ".format(accuracy, round(percentual)))

#Criando a matriz de cofusão 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt 
cm = confusion_matrix(y_test,previsao) #gera a matrix de confusão
df_cm = pd.DataFrame(cm, index= [i for i in "01234"], columns=[i for i in "01234"]) # cria o df com as classes
plt.figure(figsize=(10,7)) #indica o tamanho da figura 
sn.heatmap (df_cm, annot= True ) #plota a figura 

#vamos escolher apenas 13 atributos para realizar previsão de doenças cardiácas 
dataset_to_array = np.array(dataset)
label = dataset_to_array[:,57]
label = label.astype("int") # transforma os valores do label em inteiros
label[label>0] = 1 #quando os dados estão 0 é saúdavel, no demais doente, estamos equalizando todos para 1 doente 
label

#dividir o teste apenas com os valores do paciente sem co
x_train, x_test, Y_train, Y_test = train_test_split (dataset_paciente, label, random_state= 223) #agora utilziando apenas os valores de 0 e 1
#criando o bjeto de SVM com o SVC
model2SVM = SVC(kernel="linear") #posso escolher o KErnel polinominal 
#criando o modelo de treinamento  
modelSVM.fit(x_train,Y_train)

#previsao do novo modelo
predict = modelSVM.predict(x_test)


#verificando a accuracy do novo modelo
accuracyLabel = accuracy_score(Y_test,predict)
perce = round(accuracyLabel*100)
print("A accuracia utilziando o SVM foi de {}, \n com um percentual de {} .".format(accuracyLabel, perce))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#criando a matrix de confusão
matrixLabel = confusion_matrix(Y_test,predict)
matrixLabel = pd.DataFrame(matrixLabel)
plt.figure(figsize=(10,7))
sn.heatmap(matrixLabel, annot=True)






