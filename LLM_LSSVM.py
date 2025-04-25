'''
Implementação da proposta Fixed Sized Levenberg-Marquardt
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
from Utils import Utils

#------------------------------------------------------------------------------
#Implementação da classe
#------------------------------------------------------------------------------
class LLM_LSSVM(Kernel):

    #Método construtor
    def __init__(self, gamma = 0.5, tau = 2, kappa = 5, Red = 0.2, eps = 2.5, kernel = "gaussiano"):
        Kernel.__init__(self, gamma, kernel)
        self.tau = tau
        self.kappa = kappa
        self.Red = Red
        self.eps = eps

    #------------------------------------------------------------------------------
    #Implementação do Método Fit() para a proposta melhorada Fixed Sized Levenberg-
    #Marquardt (FSLM-LSSVM)
    #------------------------------------------------------------------------------
    def fit(self, X, y, mu = 0.5, N = 100, epsilon = 0.001):
        """
        Interface do método
        Ação: Este método visa realizar a poda iterativa dos vetores
        de suporte estimados no LSSVM, trabalhando com a matriz dos
        coeficientes completa em todas as iterações.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        mu: Taxa de aprendizado (escalar);
        kappa: Termo que define a faixa de poda (escalar);
        Red: Percentual de remoção (escalar);
        N: Número máximo de iterações (escalar);
        kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
        #tau: termo de regularização do problema primal do LSSVM (escalar).
        #epsilon: Tolerância (Critério de parada)

        OUTPUT:
        vetor esparso de multiplicadores de Lagrange estimados;
        Erro Quadrático Médio (MSE) a cada iteração;
        Índices dos Vetores de Suporte para cada iteração.
        """
        
        #Obtenção do B para todo o dataset
        B_total = Kernel.Construct_A_B(X, y, self.kernel, self.gamma, self.tau)['B']
        
        #Realizando um procedimento de clusterização utilizando o DBSCAN
        #e recuperando os índices do cluster mais ímpuro
        impurity_index = Utils.cluster_optimum(X, y, self.eps)
        
        #Obtendo os índices do complemento do cluster mais ímpuro
        purity_index = np.array(list(set(range(X.shape[0])).difference(set(impurity_index))))
        purity_index = np.squeeze(purity_index)
        
        #Construção da matriz A e vetor b e inicialização aleatória
        #do vetor de multiplicadores de Lagrange
        A = Kernel.Construct_A_B(X[purity_index, :], y[purity_index], self.kernel, self.gamma, self.tau)['A']
        B = Kernel.Construct_A_B(X[purity_index, :], y[purity_index], self.kernel, self.gamma, self.tau)['B']
        
        A_impurity = Kernel.Construct_A_B(X[impurity_index, :], y[impurity_index], self.kernel, self.gamma, self.tau)['A']
        B_impurity = Kernel.Construct_A_B(X[impurity_index, :], y[impurity_index], self.kernel, self.gamma, self.tau)['B']
        
        #Obtenção do número de amostras
        n_samples, n_features = X[purity_index, :].shape
        
        #Inicialização aleatória do vetor de multiplicadores de Lagrange
        z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples))
        z = pd.Series(z_inicial, index=purity_index)
        A = pd.DataFrame(A, index = purity_index, columns = purity_index)
        
        #Obtenção dos multiplicadores de lagrange para o cluster mais impuro
        z_target = np.zeros(X.shape[0])
        z_impurity = ln.pinv(A_impurity).dot(B_impurity)
        z_target[impurity_index] = np.squeeze(z_impurity)
        
        #Obtenção do A_target
        A_target = np.zeros((X.shape[0], X.shape[0]))
        A_target[np.ix_((impurity_index), (impurity_index))] = A_impurity

        #Listas para armazenamento
        erros = []
        idx_suporte = []
        
        #Loop de iteração
        for k in range(1, N + 1):

            #Calculando o erro associado
            erro = np.squeeze(B) - np.matmul(A, z)
            erro_target = np.squeeze(B_total) - np.matmul(A_target, z_target)

            #Atualização
            z_anterior = z
            z = z + np.matmul(ln.inv(np.matmul(A.T, A) + mu * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

            #Condição para manter a atualização
            erro_novo = np.squeeze(B) - np.matmul(A, z)
            if np.mean(erro_novo**2) < np.mean(erro**2):
                z = z
            else:
                mu = mu/10
                z = z_anterior
            
            #Condição para a janela de poda
            if k > self.kappa and k < N - self.kappa:
                
                #Realização da poda
                n_colunas_removidas = int((n_samples - (n_samples * self.Red))/(N - (2 * self.kappa)))

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                #idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]
                sorted_indices = np.abs(z).sort_values(ascending = False).index
                idx_remover = sorted_indices[:n_colunas_removidas]

                #Adição dos multiplicadores de lagrange
                z_target[idx_remover] = z[idx_remover]
                
                #Adição das colunas em A_target
                A_target[np.ix_((purity_index), (idx_remover))] = A.loc[purity_index, idx_remover]

                #índices dos vetores de suporte
                impurity_index = np.append(impurity_index, (idx_remover))
                idx_suporte.append(impurity_index)
                
                #Remoção das colunas de A e linhas de z
                A = A.drop(idx_remover, axis = 1)
                #z = np.delete(z, idx_remover, axis = 0)
                z = z.drop(idx_remover)
                
                #Atualização dos indices puros
                purity_index = z.index

            #Outra condição
            if k == N - self.kappa:

                #Realização da poda
                n_colunas_removidas = int(((n_samples - (n_samples * self.Red))/(N - (2 * self.kappa))) + ((n_samples - n_samples * self.Red)%(N - (2 * self.kappa))))

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                #idx_remover = np.argsort(-np.abs(np.squeeze(z)))[:n_colunas_removidas-1]
                sorted_indices = np.abs(z).sort_values(ascending = False).index
                idx_remover = sorted_indices[:n_colunas_removidas]

                #Adição dos multiplicadores de lagrange
                z_target[idx_remover] = z[idx_remover]
                
                #Adição das colunas em A_target
                A_target[np.ix_((purity_index), (idx_remover))] = A.loc[purity_index, idx_remover]

                #índices dos vetores de suporte
                impurity_index = np.append(impurity_index, (idx_remover))
                idx_suporte.append(impurity_index)
                
                #Remoção das colunas de A e linhas de z
                A = A.drop(idx_remover, axis = 1)
                #z = np.delete(z, idx_remover, axis = 0)
                z = z.drop(idx_remover)
                
                #Atualização dos indices puros
                purity_index = z.index
            
            #Armazenando o erro
            erro_novo_target = np.squeeze(B_total) - np.matmul(A_target, z_target)
            erros.append(np.mean(erro_target**2))
            
            #Critério de parada
            if np.abs(np.mean(erro_novo_target**2)) < epsilon:
                break
            
        mult_lagrange = z_target

        #Resultados
        resultados = {"mult_lagrange": mult_lagrange,
                        "Erros": erros,
                        "Indices_multiplicadores": idx_suporte,
                        "Iteração": k}

        #Retornando os multiplicadores de Lagrange finais
        return(resultados)
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para a primeira proposta considerando um
    #problema de classificação
    #------------------------------------------------------------------------------
    def predict(self, alphas, X_treino, X_teste):
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
            estimado[i] = np.sign(np.dot(np.squeeze(alphas), K[i]))
        
        return estimado
    
    
    