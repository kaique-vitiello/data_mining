#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Cobrança
## Autor: Trabalho em grupo
##------------------------------------------------------------------------

# Bibliotecas padrão
import DM_Utils as du
import numpy as np
import pandas as pd

## Carregando os dados
dataset = pd.read_csv('/home/kvl/git/data_mining/db/BASE.txt',sep='\t') # Separador TAB

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis
#------------------------------------------------------------------------------------------
dataset['PRE_DIAS_P'] = [1 if np.isnan(x) or x > 60 else x/60 for x in dataset['DIAS_PRIMEIRA_PARCELA']] 
dataset['PRE_IDADE'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] 
dataset['PRE_IDADE'] = [1 if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] 
dataset['PRE_QTDE_DIVIDAS'] = [0 if np.isnan(x) else x/16. for x in dataset['QTDE_DIVIDAS_ANTERIORES']]
dataset['PRE_SEXO_M'] = [1 if x=='M' else 0 for x in dataset['CD_SEXO']]
dataset['PRE_CD_B1'] = [1 if x== 'B1'else 0 for x in dataset['CD_BANCO']]
dataset['PRE_CD_B237'] = [1 if x=='B237' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B341'] = [1 if x=='B341' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B33'] = [1 if x=='B33' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B104'] = [1 if x=='B104' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B156'] = [1 if x=='B156' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B399'] = [1 if x=='B399' else 0 for x in dataset ['CD_BANCO']]
dataset['PRE_CD_B47'] = [1 if x=='B47' else 0 for x in dataset ['CD_BANCO']]
dataset['pre0-9'] = [1 if np.isnan(x) or x <= 9 else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre10-19'] = [1 if x>9 or x <=19  else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre20-29'] = [1 if x>19 or x <=29  else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre30-39'] = [1 if x>29 or x <=39  else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre40-49'] = [1 if x>39 or x <=49  else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre50-59'] = [1 if x>49 or x <=59  else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['pre60'] = [1 if x>59 else 0 for x in dataset['DIAS_PRIMEIRA_PARCELA']]
dataset['PRE_NOVO'] = [1 if x=='NOVO' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_NEUTRO'] = [1 if x=='NEUTRO' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_EXPERIENTE'] = [1 if x=='EXPERIENTE' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_VIP'] = [1 if x=='VIP' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_CDG'] = [1 if x=='G' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDD'] = [1 if x=='D' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDE'] = [1 if x=='E' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDF'] = [1 if x=='F' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDC'] = [1 if x=='C' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDA'] = [1 if x=='A' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_CDB'] = [1 if x=='B' else 0 for x in dataset['CD_CARTEIRA_CLIENTE']]
dataset['PRE_GRUPO_1'] = [1 if x=='ATÉ 1 ANO' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PRE_GRUPO_2'] = [1 if x=='ENTRE 1 E 2 ANOS' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PRE_GRUPO_3'] = [1 if x=='ENTRE 2 E 4 ANOS' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PRE_GRUPO_4'] = [1 if x=='ENTRE 4 E 9 ANOS' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PRE_GRUPO_5'] = [1 if x=='ENTRE 9 E 14 ANOS' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PRE_GRUPO_6'] = [1 if x=='MAIOR OU IGUAL A 15' else 0 for x in dataset['GRUPO_IDADE_CONTA']]
dataset['PREVALOR0'] = [1 if np.isnan(x) or x <= 300 else 0 for x in dataset['VALOR']]
dataset['PREVALOR300'] = [1 if x>300 or x <=600  else 0 for x in dataset['VALOR']]
dataset['PREVALOR600'] = [1 if x>600 or x <=900  else 0 for x in dataset['VALOR']]
dataset['PREVALOR900'] = [1 if x>900 or x <=1200  else 0 for x in dataset['VALOR']]
dataset['PREVALOR1200'] = [1 if x>1200 else 0 for x in dataset['VALOR']]
dataset['PRE_ALVO_1_0'] = [0 if np.isnan(x) else x for x in dataset['ALVO_1_0']]

##---------------------------------------------------------------------------
# Selecionando as colunasjá pré-processadas
# ---------------------------------------------------------------------------
cols_in =  ['PRE_DIAS_P',
            'PRE_IDADE',
            'PRE_QTDE_DIVIDAS',
            'PRE_SEXO_M',
            'PRE_CD_B1',
            'PRE_CD_B237',
            'PRE_CD_B341',
            'PRE_CD_B33',
            'PRE_CD_B104',
            'PRE_CD_B156',
            'PRE_CD_B399',
            'PRE_CD_B47',
            'pre0-9',
            'pre10-19',
            'pre20-29',
            'pre30-39',
            'pre40-49',
            'pre50-59',
            'pre60',
            'PRE_NOVO',
            'PRE_NEUTRO',
            'PRE_EXPERIENTE',
            'PRE_VIP',
            'PRE_CDG',
            'PRE_CDD',
            'PRE_CDE',
            'PRE_CDF',
            'PRE_CDC',
            'PRE_CDA',
            'PRE_CDB',
            'PRE_GRUPO_1',
            'PRE_GRUPO_2',
            'PRE_GRUPO_3',
            'PRE_GRUPO_4',
            'PRE_GRUPO_5',
            'PRE_GRUPO_6',
            'PREVALOR0',
            'PREVALOR300',
            'PREVALOR600',
            'PREVALOR900',
            'PREVALOR1200']
            #'PRE_ALVO_1_0'] # Com ALVO temporariamente

## Exportando os dados pre-processados
dataset.to_csv('resultado_preproc.csv')

## Separando em dados de treinamento e teste com Oversampling
##------------------------------------------------------------
y = dataset['PRE_ALVO_1_0']
X = dataset[cols_in] # Com ALVO temporariamente
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 125)

#---------------------------------------------------------------------------
## Selecionando Atributos com RFE - Recursive Feature Elimination
#---------------------------------------------------------------------------
# feature extraction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='newton-cg')
selected = RFE(model,step=1,n_features_to_select=20).fit(X_train, y_train)

print()
print('----------------     SELEÇÃO DE VARIÁVEIS--------------------------------------')
print("Num Features: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))
print('-------------------------------------------------------------------------------')

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas por RFE
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas por RFE
#---------------------------------------------------------------------------
## Ajustando modelos - Aprendizado supervisionado  
#---------------------------------------------------------------------------
# Árvore de decisão com dados de treinamento
from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=30, min_samples_split=30,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=0, splitter='best')
dtree.fit(X_train, y_train)

# Regressão linear com dados de treinamento
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression(fit_intercept=True)
LinearReg.fit(X_train, y_train)

# Regressão logística com dados de treinamento
from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression(solver='newton-cg')
LogisticReg.fit(X_train, y_train)

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(25), learning_rate='constant',
       learning_rate_init=0.01, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.25, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train)


#---------------------------------------------------------------------------
## Salvando os Modelos para Implnatar com outro código Python
#---------------------------------------------------------------------------
du.SaveModel(dtree,'dtree')
du.SaveModel(LinearReg,'LinearReg')
du.SaveModel(LogisticReg,'LogisticReg')
du.SaveModel(RNA,'RNA')

#---------------------------------------------------------------------------
## Previsão treinamento e teste - CLASSIFICAÇÃO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)
# Regressão Linear
y_pred_train_RL = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_train)] )
y_pred_test_RL  = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_test)])
# Regressão Logística
y_pred_train_RLog = LogisticReg.predict(X_train)
y_pred_test_RLog  = LogisticReg.predict(X_test)
# Redes Neurais
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA = RNA.predict(X_test)


#---------------------------------------------------------------------------
## Cálcula e mostra a Acurácia dos modelos
#---------------------------------------------------------------------------
from sklearn import metrics
print()
print('----------------------------------------------------------------------------')
print('----------     ACURÁCIA     ------------------------------------------------')
print('----------------------------------------------------------------------------')
print('Acurácia Árvore de Decisão:   ',metrics.accuracy_score(y_test, y_pred_test_DT))
print('Acurácia Regressão Linear:    ',metrics.accuracy_score(y_test, y_pred_test_RL))
print('Acurácia Regressão Logística: ',metrics.accuracy_score(y_test, y_pred_test_RLog))
print('Acurácia Redes Neurais:       ',metrics.accuracy_score(y_test, y_pred_test_RNA))
print('----------------------------------------------------------------------------')
print()
#---------------------------------------------------------------------------
## Mostra a Acurácia dos modelos
#---------------------------------------------------------------------------
print()
print('----------------------------------------------------------------------------')
print('----------     MATRIZ DE CONFUSÃO    ---------------------------------------')
print('----------------------------------------------------------------------------')
print('--  Árvore de Decisão  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_DT, rownames=['Real'], colnames=['Predito'], margins=True))
print('----------------------------------------------------------------------------')
print('--  Regressão Linear  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RL, rownames=['Real'], colnames=['Predito'], margins=True))
print('----------------------------------------------------------------------------')
print('--  Regressão Logística  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RLog, rownames=['Real'], colnames=['Predito'], margins=True))
print('----------------------------------------------------------------------------')
print('--  Redes Neurais  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RNA, rownames=['Real'], colnames=['Predito'], margins=True))
print('----------------------------------------------------------------------------')
print()
print(y_pred_test_RNA)
#---------------------------------------------------------------------------
## Previsão treinamento e teste - REGRESSÂO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT_R  = dtree.predict_proba(X_train)[:,1]
y_pred_test_DT_R  = dtree.predict_proba(X_test)[:,1]

# Regressão Linear
y_pred_train_RL_R = LinearReg.predict(X_train)
y_pred_test_RL_R  = LinearReg.predict(X_test)
# Regressão Logística
y_pred_train_RLog_R = LogisticReg.predict_proba(X_train)[:,1]
y_pred_test_RLog_R  = LogisticReg.predict_proba(X_test)[:,1]
# Redes Neurais
y_pred_train_RNA_R = RNA.predict_proba(X_train)[:,1]
y_pred_test_RNA_R  = RNA.predict_proba(X_test)[:,1]

#---------------------------------------------------------------------------
## Cálcula e mostra RMSE dos modelos
#---------------------------------------------------------------------------

from math import sqrt
print()
print('----------------------------------------------------------------------------')
print('----------     RMSE ERROR    -----------------------------------------------')
print('----------------------------------------------------------------------------')
print('Árvore de Decisão:  ',  sqrt(np.mean((y_test - y_pred_test_DT_R) **2) ))
print('Regressão Linear:   ',  sqrt(np.mean((y_pred_test_RL_R -  y_test) ** 2) ))
print('Regressão Logística:',  np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5)
print('Redes Neurais:      ',  np.mean((y_pred_test_RNA_R - y_test) ** 2) ** 0.5)
print('----------------------------------------------------------------------------')
print()

#---------------------------------------------------------------------------
## Cálcula o KS2
#---------------------------------------------------------------------------
print()
print('----------------------------------------------------------------------------')
print('----------------     KS2    ------------------------------------------------')
print('----------------------------------------------------------------------------')
print('Árvore de Decisão:   ',du.KS2(y_test,y_pred_test_DT_R))
print('Regressão Linear:    ',du.KS2(y_test,y_pred_test_RL_R))
print('Regressão Logística: ',du.KS2(y_test,y_pred_test_RLog_R))
print('Redes Neurais:       ',du.KS2(y_test,y_pred_test_RNA_R))
print('----------------------------------------------------------------------------')
print()

#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT_R, columns=['REGRESSION_DT'])
df_train['CLASSIF_DT'] = y_pred_train_DT
df_train['REGRESSION_RL'] = y_pred_train_RL_R
df_train['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_train_RL]
df_train['REGRESSION_RLog'] = y_pred_train_RLog_R
df_train['CLASSIF_RLog'] = y_pred_train_RLog
df_train['REGRESSION_RNA'] = y_pred_train_RNA_R
df_train['CLASSIF_RNA'] = y_pred_train_RNA
df_train['ALVO'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'

# Conjunto de test
df_test = pd.DataFrame(y_pred_test_DT_R, columns=['REGRESSION_DT'])
df_test['CLASSIF_DT'] = y_pred_test_DT
df_test['REGRESSION_RL'] = y_pred_test_RL_R
df_test['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_test_RL]
df_test['REGRESSION_RLog'] = y_pred_test_RLog_R
df_test['CLASSIF_RLog'] = y_pred_test_RLog
df_test['REGRESSION_RNA'] = y_pred_test_RNA_R
df_test['CLASSIF_RNA'] = y_pred_test_RNA
df_test['ALVO'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 

print()
print('----------------    INÍCIO DA EXPORTAÇÃO RESULTADOS   ----------------------------------')
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train], sort = False)

## Exportando os dados para avaliação dos resultados em outra ferramenta
df_total.to_csv('resultado_comparacao.csv')
print('----------------     FIM DA EXPORTAÇÃO RESULTADOS   ------------------------------------')