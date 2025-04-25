'''
Implementação da classe para a construção das funções de kernel
Data:15/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from numpy import linalg

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class Kernel:

    #Método Construtor
    def __init__(self, gamma, C = 1, d = 3, kernel = "gaussiano"):
        self.gamma = gamma
        self.C = C
        self.d = d
        self.kernel = kernel
    
    #Métodos Estáticos para as funções de kernel
    @staticmethod
    def linear_kernel(x, x_k):
        return np.dot(x, x_k)
    
    @staticmethod
    def polinomial_kernel(self, x, y):
        return (np.dot(x, y) + self.C)**self.d
    
    @staticmethod
    def gaussiano_kernel(gamma, x, y):
        return np.exp(-gamma * linalg.norm(x - y)**2)
    
    @staticmethod
    def Construct_A_B(X, y, kernel, gamma, tau):
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
                    K[i, j] = Kernel.linear_kernel(X[i], X[j])
                
                if kernel == "gaussiano":
                    K[i, j] = Kernel.gaussiano_kernel(gamma, X[i], X[j])
                
                if kernel == "polinomial":
                    K[i, j] = Kernel.polinomial_kernel(X[i], X[j])
        
        #--------------------------------------------------------------------------
        #Decomposição da matriz de kernel
        #--------------------------------------------------------------------------
        #Cholesky Incompleta
        P = np.linalg.cholesky(K + 0.01 * np.diag(np.full(K.shape[0], 1)))
        
        #Construção da matriz dos coeficiente A
        A = P.T
        
        #Construção do vetor de coeficientes b
        B = np.dot(linalg.inv(tau * np.identity(n_samples) + np.dot(P, P.T)), np.dot(P.T, y))
        B = np.expand_dims(B, axis = 1)
        
        #Resultados
        resultados = {'A': A,
                    "B": B}
        
        return(resultados)