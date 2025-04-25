'''
Implementação da classe para os métodos fit e predict
para a proposta seminal de Suykes P_LSSVM
Data: 16/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from numpy import linalg as ln
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Kernel import Kernel
from Utils import Utils

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class P_LSSVM(Kernel):

    #Método construtor
    def __init__(self, gamma = 0.5, tau = 0.5, n_drop = 10, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.tau = tau
        self.n_drop = n_drop

    #------------------------------------------------------------------------------
    #Proposta de poda iterativa sugerida por Johan Suykens:
    #Source: SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector
    #machine classifiers. Neural processing letters, v. 9, p. 293-300, 1999.
    #------------------------------------------------------------------------------
    def fit(self, X, y, N = 30, epsilon = 0.01):
        '''
        Método para ajustar o procedimento de poda apresentado por Suykens em seu
        trabalho seminal.
        INPUT:
            X - Array de features (array de dimensão n x p);
            y - Array de targets (array de dimensão n x 1);
            tau - Parâmetro de regularização para o problema primal do LSSVM (float);
            kernel - String indicando o tipo de função de kernel ("linear",
                                                                "polinomial",
                                                                "gaussiano");
            n_drop - Inteiro indicando a quantidade de vetores de suporte removidos
            a cada iteração (int);
            N - Número máximo de iterações (int);
            epsilon - Tolerância para a redução da performamce (float).
            
        OUTPUT:
            Array esparso de Multiplicadores de Lagrange ótimos.
            
        '''
        
        #Conversão para dataframe
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        #Divisão Treino/Validação
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, 
                                                            random_state = 42)
        
        y_train = pd.Series(y_train)
        y_valid = pd.Series(y_valid)
        
        #Inicialização
        performance = []
        
        #Loop principal
        for i in range(N):
            
            #Treinando no conjunto de treinamento
            A = Utils.Construct_A_b(X_train, y_train, self.kernel, self.tau, self.gamma)['A']
            B = Utils.Construct_A_b(X_train, y_train, self.kernel, self.tau, self.gamma)['B']
            
            #Aplicação dO Método da pseudoinversa para a solução do sistema Ax = B
            solution = np.squeeze(np.dot(ln.pinv(A), B))
            b = np.squeeze(solution[0])
            alphas = np.squeeze(solution[1:])
            y_hat = Utils.predict_class(alphas, b, self.gamma, self.kernel, X_train, y_train, X_valid)
            acuracia = accuracy_score(y_valid, y_hat)
            
            #Realizando a poda dos multiplicadores de Lagrange
            solution = pd.Series(solution[1:], index = X_train.index)
            sorted_indices = np.abs(solution).sort_values(ascending = False).index
            idx_remover = sorted_indices[:self.n_drop]
            idx_continua = sorted_indices[self.n_drop:]
            
            #Removendo os multiplicadores de lagrange e as colunas correspondentes
            #no array de features
            solution = solution.drop(idx_remover, axis = 0)
            X_train = X_train.drop(idx_remover, axis = 0)
            y_train = y_train.drop(idx_remover, axis = 0)
            
            #Avaliação do critério de parada
            alphas = np.squeeze(solution)
            y_hat = Utils.predict_class(alphas, b, self.gamma, self.kernel, X_train, y_train, X_valid)
            acuracia_novo = accuracy_score(y_valid, y_hat)
            performance.append(acuracia_novo)
            difference = acuracia - acuracia_novo
            if difference < epsilon:
                break
        
        alphas = np.zeros(X.shape[0])
        alphas[idx_continua] = solution
        resultado = {'b':b,
                    "mult_lagrange": alphas,
                    "métrica": performance,
                    'diferença': difference,
                    'iteração': i}
        
        return resultado
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para problemas de classificação e regressão
    #utilizando o CG de Hestenes-Stiefel
    #fontes: 
    #    -SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector machine classifiers.
    #     Neural processing letters, v. 9, p. 293-300, 1999.    
    #
    #    -GOLUB, Gene H.; VAN LOAN, Charles F. Matrix computations. JHU press, 2013.   
    #------------------------------------------------------------------------------
    def predict(self, alphas, b, X_treino, y_treino, X_teste):
        #Inicialização
        estimado = np.zeros(X_teste.shape[0])
        n_samples_treino = X_treino.shape[0]
        n_samples_teste = X_teste.shape[0]
        K = np.zeros((n_samples_teste, n_samples_treino))
    
        #Construção da matriz de Kernel
        for i in range(n_samples_teste):
            for j in range(n_samples_treino):
                
                if self.kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X_teste[i], X_treino[j])
                
                if self.kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X_teste[i], X_treino[j])
                
                if self.kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(self.gamma, X_teste[i], X_treino[j])
                
            #Realização da predição
            estimado[i] = np.sign(np.sum(np.multiply(np.multiply(alphas, y_treino), K[i])) + b)
        
        return estimado

