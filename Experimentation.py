'''
Script para realização das etapas de carregamento de dataset,
divisão treino/validação/teste, tuning dos modelos e hold-out
para obtenção das métricas de performance
Data: 29/10/2024
'''

#--------------------------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#--------------------------------------------------------------------------------------------------
#Manipulação de dados
import pandas as pd
import numpy as np
from numpy.linalg import cholesky

#Visualização gráfica
import matplotlib.pyplot as plt
import seaborn as sns

#DS e Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.metrics import accuracy_score

#Proposta
from LLM_LSSVM import LLM_LSSVM

#Modelos para comparação
##from P_LSSVM import fit_P_LSSVM
from LSSVM import LSSVM
from P_LSSVM import P_LSSVM
from IP_LSSVM import IP_LSSVM

#Tuning dos modelos
import optuna
from optuna_dashboard import run_server

#Tempo de processamento
from time import process_time

#Performance
from métricas import metricas

#Outros
import os
import pathlib

#--------------------------------------------------------------------------------------------------
#Implementando algumas funções relevantes
#--------------------------------------------------------------------------------------------------
#Carregando dataset
def load_data(data_path, name_dataset):
    '''
    Função para carregar os datasets em um pandas
    Dataframe, automatizando o processo de obtenção
    das performances dos modelos
    INPUT:
    data_path: Caminho indicando o diretório do dataset;
    name_dataset: Nome do dataset (string)

    OUTPUT:
    pandas Dataframe com os dados considerados. 
    '''
    extension = pathlib.Path(name_dataset).suffix
    name_dataset = pathlib.Path(name_dataset).stem
    csv_path = os.path.join(data_path, f"{name_dataset}{extension}")
    return pd.read_csv(csv_path)


#Remoção de registros com valores nulos
def drop_nan(dataset):
    '''
    Remove registros (linhas) com valores nulos
    INPUT:
    dataset: Dataframe com os dados de entrada (Dataframe);

    OUTPUT:
    Dataframe sem as linhas com valores nulos.
    '''

    return dataset.dropna(axis = 0)

#Análise Univariada
def filtro_volumetria(df):
    '''
    função para remover features com um percentual de nulos
    acima do valor do threshold
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volumetria
    '''
    
    for column in df.columns:
        if df[column].isna().sum()/df.shape[0] > 0.8:
            df = df.drop([column], axis = 1)
    
    return df

def filtro_volatilidade(df):
    '''
    função para remover features com baixa variabilidade, que por
    sua vez, não contribuem significativamente para a discriminação
    interclasses
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volatilidade
    '''
    
    for column in df.select_dtypes(include = ["int64", "float64"]).columns.values:
        if round(df[column].var(), 2) < 0.9:
            df = df.drop([column], axis = 1)
    return(df)

def filtro_correlacao(df):
    '''
    função para remover features colineares com base na correlação de spearmen
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features descorrelacionadas
    '''

    # Calcula a matriz de correlação de spearmen
    corr_matrix = df.corr(method = 'spearman')
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= 0.7:
    
                drop_cols.append(col.values[0])

    
    drops = set(drop_cols)
    df = df.drop(columns=drops)
    print('Coluna removida {}'.format(drops))
    return df

#--------------------------------------------------------------------------------------------------
#Implementando o pipeline de dados
#-------------------------------------------------------------------------------------------------- 
#Inicialização do caminho para os arquivos csv's
data_path_datasets = '/Users/Felipe/Documents/Python Scripts/Tese/Datasets'
data_path_predictions = '/Users/Felipe/Documents/Python Scripts/Tese/Predictions'
data_path_performances = '/Users/Felipe/Documents/Python Scripts/Tese/Performances'
data_path_times = '/Users/Felipe/Documents/Python Scripts/Tese/Times'

#Inicialização dos nomes dos datasets
'breast.csv', 'column_2C.csv', 'australian.csv', 'bupa.csv', 'haberman.csv',
'ionosphere.csv','diabetes.csv','SouthGermanCredit.csv', 'australian.csv'
datasets = ['SouthGermanCredit.csv']

#Definição da função para inserção no pipeline
drop_nan_pipeline = FunctionTransformer(drop_nan)
volumetria = FunctionTransformer(filtro_volumetria)
volatilidade = FunctionTransformer(filtro_volatilidade)
correlacao = FunctionTransformer(filtro_correlacao)

#Definição de um pipeline de dados
data_pipeline = Pipeline([('drop_nan', drop_nan_pipeline),
                          #('Volumetria', volumetria),
                          #('Volatilidade', volatilidade),
                          #('Correlação', correlacao),
                          ('Normalização', StandardScaler())])
                          

#--------------------------------------------------------------------------
#Tuning dos modelos propostos
#--------------------------------------------------------------------------
#Definição da função objetivo
def objective(trial):
        
        classifier_name = k

        if classifier_name == 'SVC':
            svr_c = trial.suggest_float('svr_c', 1e-2, 1e10, log=True)
            classifier_obj = SVC(C=svr_c)
            
        if (classifier_name == 'LSSVM'):
            param_gamma = trial.suggest_float('param_gamma', 0.1, 1, log=True)
            param_tau = trial.suggest_float('param_tau', 1, 10, log=True)
            classifier_obj = LSSVM(gamma = param_gamma, tau = param_tau)

        if (classifier_name == 'LLM_LSSVM'):
            param_gamma = trial.suggest_float('param_gamma', 0.1, 1, log=True)
            param_tau = trial.suggest_float('param_tau', 1, 10, log=True)
            classifier_obj = LLM_LSSVM(gamma = param_gamma, tau = param_tau)
            
        if (classifier_name == 'P_LSSVM'):
            param_gamma = trial.suggest_float('param_gamma', 1e-2, 1e2, log=True)
            param_tau = trial.suggest_float('param_tau', 1e-2, 1e2, log=True)
            classifier_obj = P_LSSVM(gamma = param_gamma, tau = param_tau)
            
        if (classifier_name == 'IP_LSSVM'):
            param_gamma = trial.suggest_float('param_gamma', 1e-2, 1, log=True)
            param_tau = trial.suggest_float('param_tau', 1, 10, log=True)
            classifier_obj = IP_LSSVM(gamma = param_gamma, tau = param_tau)
            
        if (classifier_name == 'LSSVM') | (classifier_name == 'P_LSSVM') | (classifier_name == 'IP_LSSVM'):
            resultados = classifier_obj.fit(X_train_processed, y_train)
            alphas = resultados['mult_lagrange']
            b = resultados['b']
            y_hat = classifier_obj.predict(alphas, b, X_train_processed, y_train, X_valid_processed)
            
        if (classifier_name == 'LLM_LSSVM'):
            resultados = classifier_obj.fit(X_train_processed, y_train)
            alphas = resultados['mult_lagrange']
            y_hat = classifier_obj.predict(alphas, X_train_processed, X_valid_processed)

        if classifier_name == 'SVC':
            classifier_obj.fit(X_train_processed, y_train)
            y_hat = classifier_obj.predict(X_valid_processed)

        score = accuracy_score(y_pred = y_hat, y_true = y_valid)

        return score


#Definição dos modelos avaliados
modelos_comparacao_sklearn = {'KNN': KNeighborsClassifier(),
                              'MLP': MLPClassifier(),
                              'Logística': LogisticRegression(),
                              'SVM': SVC()}

#---------------------------------------------------------------------------
#Implementando uma abordagem baseada em Hold-Out com 20 realizações para
#obtenção dos resultados
#---------------------------------------------------------------------------
for dataset in datasets:
        #Carregando a base
        df = load_data(data_path_datasets, dataset)

        #Separando features de target
        X = df.iloc[:, range(df.shape[1]-1)]
        y = np.array(df.iloc[:, df.shape[1] -1])

        #Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

        #Divisão subtreino/validação
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3,
                                                               random_state = 42)

        #Aplicando o pipeline de dados ao conjunto de treino
        X_train_processed = np.array(data_pipeline.fit_transform(X_train))

        #Aplicando o pipeline de dados ao conjunto de validação
        X_valid_processed = np.array(data_pipeline.fit_transform(X_valid))

        #Aplicando o pipeline de dados ao conjunto de teste
        X_test_processed = np.array(data_pipeline.fit_transform(X_test))
        
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #Tuning dos modelos
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        best_params = {}
        modelos = ['SVC',
                   'LSSVM',
                   'LLM_LSSVM',
                   'IP_LSSVM', 
                   'P_LSSVM']
        
        for k in modelos:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(direction = 'maximize',
                                        storage = storage, 
                                        study_name = f"tuning_model {k}")
            study.optimize(objective, n_trials = 100)
            run_server(storage)

            best_params[k] = study.best_params

        #Passando para um dataframe e esportando
        best_params_df = pd.DataFrame(best_params)
        best_params_df.to_excel(f"best_params_{dataset}.xlsx")

        
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #Obtendo resultados para os modelos do sklearn
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #Inicializando os modelos com seus hiperparâmetros ótimos
        modelos_primal = {'LSSVM': LSSVM(tau = best_params['LSSVM']['param_tau'],
                                         gamma = best_params['LSSVM']['param_gamma']),
                                        'P_LSSVM': P_LSSVM(tau = best_params['P_LSSVM']['param_tau'],
                                         gamma = best_params['P_LSSVM']['param_gamma']),
                                        'IP_LSSVM': IP_LSSVM(tau = best_params['IP_LSSVM']['param_tau'],
                                         gamma = best_params['IP_LSSVM']['param_gamma'])}

        
        modelos_dual = {'LLM_LSSVM': LLM_LSSVM(tau = best_params['FSLM_LSSVM_improved']['param_tau'],
                                         gamma = best_params['FSLM_LSSVM_improved']['param_gamma'])}
        
        
        #Monte carlo com 20 realizações
        for i in range(20):

            #Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
            
            #Aplicando o pipeline de dados ao conjunto de treino
            X_train_processed = np.array(data_pipeline.fit_transform(X_train))
            
            #Aplicando o pipeline de dados ao conjunto de teste
            X_test_processed = np.array(data_pipeline.fit_transform(X_test))

            #Ajuste dos modelos
            train_predict, test_predict = {}, {}
            time_processing = {}
            performance = {}
            
            for k in modelos_comparacao_sklearn:

                tic = process_time() #Inicialização do contador
                modelos_comparacao_sklearn[k].fit(X_train_processed, y_train)
                toc = process_time() #Finalização do contador

                #Armazenando as performance e os tempos de treinamento
                time_processing[k] = toc - tic
                train_predict[k] = modelos_comparacao_sklearn[k].predict(X_train_processed)
                test_predict[k] = modelos_comparacao_sklearn[k].predict(X_test_processed)

                #Obtendo as métricas  de performance
                performance[k] = metricas(y_test, test_predict[k])
            
            
            for k in modelos_primal:

                tic = process_time() #Inicialização do contador
                resultados = modelos_primal[k].fit(X_train_processed, y_train)
                toc = process_time() #Finalização do contador

                #Armazenando as performance e os tempos de treinamento
                time_processing[k] = toc - tic

                alphas = resultados['mult_lagrange']
                b = resultados['b']
                train_predict[k] = modelos_primal[k].predict(alphas, b, X_train_processed,
                                                            y_train, X_train_processed)
                test_predict[k] = modelos_primal[k].predict(alphas, b, X_train_processed,
                                                            y_train, X_test_processed)

                #Obtendo as métricas  de performance
                performance[k] = metricas(y_test, test_predict[k])
            
            
            for k in modelos_dual:

                tic = process_time() #Inicialização do contador
                resultados = modelos_dual[k].fit(X_train_processed, y_train)
                toc = process_time() #Finalização do contador

                #Armazenando as performance e os tempos de treinamento
                time_processing[k] = toc - tic

                alphas = resultados['mult_lagrange']
                train_predict[k] = modelos_dual[k].predict(alphas, X_train_processed, X_train_processed)
                test_predict[k] = modelos_dual[k].predict(alphas, X_train_processed, X_test_processed)

                #Obtendo as métricas  de performance
                performance[k] = metricas(y_test, test_predict[k])
            

            #Gerando os dataframes
            train_predict = pd.DataFrame(train_predict)
            test_predict = pd.DataFrame(test_predict)
            performance = pd.DataFrame(performance)
            time_processing = pd.Series(time_processing)

            #Exportando os resultados como excel
            train_predict.to_excel(f"train_predict_{dataset}_{i}.xlsx")
            test_predict.to_excel(f"test_predict_{dataset}_{i}.xlsx")
            performance.to_excel(f"performance_{dataset}_{i}.xlsx")
            time_processing.to_excel(f"time_processing_{dataset}_{i}.xlsx")