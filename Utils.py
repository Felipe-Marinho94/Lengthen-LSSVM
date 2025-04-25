'''
Classe contendo algumas fuções úteis no desenvolvimento de outras classes
Data:15/11/2024
'''
#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from numpy import linalg as ln
from sklearn.cluster import DBSCAN
from Kernel import Kernel
from math import sqrt, log
from numpy import dot

#------------------------------------------------------------------------------
#Implementação da classe
#------------------------------------------------------------------------------
class Utils(Kernel):

    #Método construtor
    def __init__(self, gamma, eps, C=1, d=3, kernel="gaussiano"):
        super().__init__(gamma, C, d, kernel)
        self.eps = eps
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def CG(A, b, epsilon, N):
        #--------------------------------------------------------------------------
        #Método iterativo para a solução de sistemas lineares Ax = B, com A sendo
        #simétrica e definida positiva
        #INPUTS:
        #A:matriz dos coeficientes de ordem m x n (array)
        #B:vetor de termos independentes m x 1 (array)
        #epsilon:tolerância (escalar)
        #N: Número máximo de iterações
        #OUTPUTS:
        #x*:vetor solucao aproximada n x 1 (array) 
        #--------------------------------------------------------------------------
        
        #Inicialização
        i = 0
        x = np.zeros(A.shape[1])
        r = b - A.dot(x)
        r_anterior = r
        
        while (np.sqrt(np.dot(r, r)) > epsilon) | (i <= N) :
            i += 1
            if i == 1:
                p = r
            else:
                beta = np.dot(r, r)/np.dot(r_anterior, r_anterior)
                p = r + beta * p
            
            lamb = np.dot(r, r)/np.dot(p, A.dot(p))
            x += lamb * p
            r_anterior = r
            r += -lamb * A.dot(p)
            
        return x
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta LLM_LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def purity_level(X, y):
        '''
        Função para determinar o nível de pureza de uma determinado conjunto com
        base nas alterações de sinal do rótulo de cada amostra
        INPUT:
            X - Array de features (Array de dimensão n x p);
            y - Array de targets (Array de dimensão n x 1);
            index - Conjunto de índices correspondentes a um subconjunto de X.
            
        OUTPUT:
            pureza - Nível de pureza dada pelas trocas de sinal no rótulo de cada
            amostra do subconjunto em análise.
        '''
        #Incialização
        contador = 0
        y_anterior = y[0]
        
        for i in range(len(y)):

            if  y[i] * y_anterior < 0:
                contador += 1
            
            y_anterior = y[i]
        
        return(contador)
    
    @staticmethod
    def cluster_optimum(X, y, eps):
        '''
        Método para determinação do cluster com maior nível de impureza, onde o 
        processo de clusterização é baseado no algoritmo DBSCAN.
        INPUT:
            X - Array de features (Array de dimensão n x p);
            y - Array de targets (Array de dimensão n x 1);
            eps - máxima distância entre duas amostras para uma ser considerada 
            vizinhança da outra (float, default = 0.5).
            
        OUTPUT:
            índices do cluster maior impureza.
        '''
        
        #Convertendo dataframe
        X = pd.DataFrame(X)
        y = pd.Series(y, name = "y")
        
        #Clusterização utilizando DBSCAN
        clustering = DBSCAN(eps = eps).fit(X)
        
        #Recuperando os índices
        cluster = pd.Series(clustering.labels_, name = "cluster")
        df = pd.concat([cluster, X, y], axis = 1)
        purity = df.groupby('cluster').apply(Utils.purity_level, df.y)
        
        return(df.where(df.cluster == purity.idxmax()).dropna(axis = 0).index)
    
    
    #---------------------------------------------------------------
    #Funções auxiliares para a proposta Prunning LSSVM
    #---------------------------------------------------------------
    @staticmethod
    def Construct_A_b(X, y, kernel, tau, gamma):
        """
        Interface do método
        Ação: Este método ecapsular o cálculo da matriz A e vetor b para
        a proposta RFSLM-LSSVM.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
        #tau: Parâmetro de regularização do problema primal do LSSVM.

        OUTPUT:
        dois array's representando a matriz A e b, respectivamente.
        """

        #Construção da matriz A e vetor b e inicialização aleatória
        #do vetor de multiplicadores de Lagrange
        n_samples, n_features = X.shape
        
        #Matriz de Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                
                #Kernel trick
                if kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X.iloc[i, :], X.iloc[j, :])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X.iloc[i, :], X.iloc[j, :])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X.iloc[i, :], X.iloc[j, :])
        
        #Construção da Matriz Omega
        Omega = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Omega[i, j] = y.iloc[i] * y.iloc[j] * K[i, j]
        
        #Construção da Matriz A
        H = Omega + (1/tau) * np.identity(n_samples)
        A = np.block([[np.array([0]), np.expand_dims(y, axis = 1).T],
                    [np.expand_dims(y, axis = 1), H]])

        #Construção do Vetor B
        B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                            np.expand_dims(np.ones(n_samples), axis = 1)), axis=0)
        
        #Resultados
        resultados = {'A': A,
                    "B": B}
        
        return(resultados)
    
    @staticmethod
    #Método para realizar a predição utilizando os multiplicadores de lagrange
    #ótimos estimados
    def predict_class(alphas, b, gamma, kernel, X_treino, y_treino, X_teste):
        #Inicialização
        alphas = np.array(alphas)
        estimado = np.zeros(X_teste.shape[0])
        n_samples_treino = X_treino.shape[0]
        n_samples_teste = X_teste.shape[0]
        K = np.zeros((n_samples_teste, n_samples_treino))
        
        #Construção da matriz de Kernel
        for i in range(n_samples_teste):
            for j in range(n_samples_treino):
                
                if kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X_teste.iloc[i, :], X_treino.iloc[j, :])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X_teste.iloc[i, :], X_treino.iloc[j, :])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X_teste.iloc[i, :], X_treino.iloc[j, :])
                
            #Realização da predição
            estimado[i] = np.sign(np.sum(np.multiply(np.multiply(alphas, y_treino), K[i])) + b)
        
        return estimado
    

