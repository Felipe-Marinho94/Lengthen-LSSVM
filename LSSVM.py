'''
Implemetação da classe para o método LSSVM
Data: 14/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from Kernel import Kernel
from Utils import Utils

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class LSSVM(Kernel):

    #Método construtor
    def __init__(self, gamma = 0.5, tau = 2, epsilon = 0.01, N = 20,  kernel = "gaussiano"):
        Kernel.__init__(self, gamma, kernel)
        self.tau = tau
        self.epsilon = epsilon
        self.N = N

    #------------------------------------------------------------------------------
    #Implementação do método fit() para problemas de classificação e regressão
    #utilizando o CG de Hestenes-Stiefel
    #fontes: 
    #    -SUYKENS, Johan AK; VANDEWALLE, Joos. Least squares support vector machine classifiers.
    #     Neural processing letters, v. 9, p. 293-300, 1999.
    #
    #    -GOLUB, Gene H.; VAN LOAN, Charles F. Matrix computations. JHU press, 2013.   
    #------------------------------------------------------------------------------
    def fit(self, X, y):
        #Inputs
        #X: array das variáveis de entrada array(n x p)
        #y: array de rótulos (classificação), variável de saída (regressão) array (n x 1)
        #tau: termo de regularização do problema primal do LSSVM (escalar)
        #kernel: string indicando o kernel utilizado ("linear", "polinomial", "gaussiano")
    
        n_samples, n_features = X.shape
    
        #Matriz de Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
            
                #Kernel trick
                if self.kernel == "linear":
                    K[i, j] = Kernel.linear_kernel(X[i], X[j])
            
                if self.kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(self.gamma, X[i], X[j])
            
                if self.kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X[i], X[j])
    
        #Construção da Matriz Omega
        Omega = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Omega[i, j] = y[i] * y[j] * K[i, j]
    
        #--------------------------------------------------------------------------
        #Construção do sistema linear com matriz dos coeficientes
        #simétrica, definda positiva: Ax = B
        #--------------------------------------------------------------------------
        #Construção da matriz A
        H = Omega + (1/self.tau) * np.identity(n_samples)
        s = np.dot(y, np.linalg.inv(H).dot(y))
        zero_linha = np.zeros((1, n_samples))
        zero_coluna = np.zeros((n_samples, 1))
        A = np.block([[s, zero_linha], [zero_coluna, H]])
    
        #Construção do vetor B
        d1 = 0
        d2 = np.expand_dims(np.ones(n_samples), axis = 1)
        b1 = np.expand_dims(np.array(np.dot(y, np.linalg.inv(H).dot(d2))), axis = 0)
        B = np.concatenate((b1, d2), axis = 0)
        B = np.squeeze(B)
    
        #Aplicação de um método iterativo para a solução do sistema Ax = B
        #guess_inicial = np.zeros(n_samples + 1)
        solution = Utils.CG(A, B, self.epsilon, self.N)
        #solution = np.linalg.inv(A).dot(B)
    
        #Obtenção do b e dos multiplicadores de Lagrange
        b = solution[0]
        alphas = solution[1:] - np.linalg.inv(H).dot(y) * b
    
        resultado = {'b': b,
                 "mult_lagrange": alphas,
                 "kernel": K,
                 'A':A,
                 'B': B}
    
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

    #------------------------------------------------------------------------------
    #Implementação dos métodos GET e SET para os atributos gamma e tau
    #------------------------------------------------------------------------------
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, valor):
        self._gamma = valor