'''
Implementação de classe com os métodos fit e predict
para a abordagem de poda IP_LSSVM
Data:16/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from numpy import linalg as ln
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Kernel import Kernel
from Utils import Utils

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class IP_LSSVM(Kernel):

    #Método construtor
    def __init__(self, gamma = 0.5, tau = 2, n_drop = 30, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.tau = tau
        self.n_drop = n_drop
    
    #------------------------------------------------------------------------------
    #Proposta de poda iterativa sugerida por CARVALHO; BRAGA, 2009:
    #A two-step sparse classifier. Pattern
    #Recognition Letters, v. 30, n. 16, p. 1507–1515, 2009
    #------------------------------------------------------------------------------
    def fit(self, X, y):
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
        
        #Construção do sitema KKT para obtenção dos multiplicadores
        A = Utils.Construct_A_b(X, y, self.kernel, self.tau, self.gamma)['A']
        B = Utils.Construct_A_b(X, y, self.kernel, self.tau, self.gamma)['B']

        #Convertendo o numpy array A para um pandas Dataframe
        #afim de facilitar a manipulação de índices
        A = pd.DataFrame(A)

        #Treinamento do LSSVM com todo o conjunto de treinamento
        solution = np.dot(ln.inv(A), B)
        alphas = solution[1:]

        #Remover os menores multiplicadores de Lagrange sem considerar
        #valor absoluto
        index_removidos = np.argsort(np.squeeze(alphas))[:self.n_drop]
        index_suporte = np.argsort(np.squeeze(alphas))[self.n_drop:]

        #Remover as colunas da matriz A correspondentes
        A = A.drop(index_removidos, axis = 1)

        #Obter uma nova solução com base na nova matriz
        solution = ln.pinv(A) @ B

        #Obtenção dos vetores suporte finais
        b = solution[0]
        alphas = np.zeros(X.shape[0])
        alphas[index_suporte] = np.squeeze(solution[1:])

        #retornando os resultados
        resultados = {'b': b,
                    "mult_lagrange": alphas}
        
        return(resultados)
    
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
