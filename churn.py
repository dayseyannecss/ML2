import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns

# buscando o dataset
churn = pd.read_csv('/home/daysesousa/archive/BankChurners.csv',usecols=list(np.arange(1,21)))



#o print seguinte imprime se o cliente deixaram de utilizar o cartão- Taxa de 16%
# será atribuido no yarget Existing Customer o valor 0 e Attrited Customer o valor 1, pois é mais fácil tratar números a string

churn['Attrition_Flag'].replace({'Existing Customer':0, 'Attrited Customer':1}, inplace=True)

print(churn['Attrition_Flag'].value_counts(normalize=True))

# categorizando por gênero, educação para entender mnelhor o problema da evasão de clientes

print(churn[['Attrition_Flag','Gender','Marital_Status']].groupby(['Gender','Marital_Status']).mean(numeric_only=True))


#corr = churn.corr(method='pearson')
# gera matriz de correlação entre variaveis
#corr = churn.corr()
#plt.figure(figsize=(12,8))
#sns.heatmap(corr, annot=True, cmap="YlGnBu")

#sns.relplot(data=churn, kind='scatter', x='Total_Trans_Amt', y='Total_Trans_Ct',hue='Attrition_Flag', height=6)

print(churn.columns)

#dividir em dados de treinamento , teste e validação

chur_full_train, churn_test = train_test_split(churn, test_size=0.2, random_state=1)
chur_train, churn_val = train_test_split(chur_full_train, test_size=0.25, random_state=1)

print(churn_test.head(5))

chur_train_dict = chur_train.to_dict(orient='records')
dv = DictVectorizer()
dv.fit(chur_train_dict)

plt.show()


