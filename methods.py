import sys
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from os import listdir
from math import sqrt

def readFile(inputs):
    shape = tuple(np.loadtxt(fname=inputs, dtype=int, delimiter=' ', max_rows=1, usecols=(0,1)))
    PreSim = np.loadtxt(fname=inputs, dtype=str, delimiter=' ', max_rows=1, usecols=(2,3))
    A = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', skiprows=1, max_rows=shape[1], usecols=np.arange(0,shape[1]))
    B = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', skiprows=shape[1], max_rows=shape[0], usecols=np.arange(0,shape[1]))

    if shape[0] != 1:
        B = np.reshape(B, (shape[0],shape[1],1))
    else:
        B = np.reshape(B, (shape[1],shape[0]))

    A = np.reshape(A, (shape[1],shape[1]))
    P = np.float64(PreSim[0])
    S = PreSim[1]

    return A, B, P, S

def readArgs():
    if len(sys.argv) > 2:
        arq = 0
        file_input = listdir('./inputs/')
        for file in file_input:
            if file == sys.argv[1]:
                arq = 'inputs/' + sys.argv[1]
        
        args = sys.argv[1:]
        print(f"Arguments count: {len(sys.argv)}")
        print(f"Arguments of the script : {args}")
        if arq == 0:
            print("Arquivo não encontrado nos Inputs ou Arquivo com nome Inválido !")
            exit(1) 
    else:
        print("Entrada de Dados Inválida !")
        exit(1)
    return arq, sys.argv[2]

def gauss(A, B):
    start = timer()

    dimensionM = A.shape[0]
    for i in np.arange(1, dimensionM):
        for j in np.arange(i, dimensionM):
            matAUX = A[j][i-1] / A[i-1][i-1]
            for k in np.arange(j,j+1):
                A[k] = A[k] - matAUX * A[i-1]
                B[k] = B[k] - matAUX * B[i-1]

    end = timer()

    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução Gauss: {timing}\n")
    return A

def subSucessiva(A, B, X):
    dimensionM = A.shape[0]

    for i in np.arange((dimensionM-1), -1, -1):
        X[i] = B[i]
        for j in np.arange((dimensionM-1), 0, -1):
            if i != j:
                X[i]= X[i] - A[i][j] * X[j]
        X[i] = X[i]/A[i][i]


def fatoraLU(A):
    start = timer()

    U = np.copy(A)
    dimensionM = A.shape[0]
    L = np.eye(dimensionM)

    for j in np.arange(dimensionM-1):
        for i in np.arange(j+1,dimensionM):
            L[i,j] = U[i,j]/U[j,j]
            for k in np.arange(j+1,dimensionM):
                U[i,k] = U[i,k] - L[i,j]*U[j,k]
            U[i,j] = 0

    end = timer()

    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução LU: {timing}\n")
    return L, U

def cholesky(A):
    start = timer()

    dimensionM = A.shape[0]
    MI = np.zeros_like(A)

    for k in np.arange(dimensionM):
        MI[k,k] = sqrt(A[k,k])
        MI[k,k+1:] = A[k,k+1:]/MI[k,k]
        for j in np.arange(k+1,dimensionM):
            A[j,j:] = A[j,j:] - MI[k,j] * MI[k,j:]
    
    end = timer()

    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução Cholesky: {timing}\n")
    
    return MI


 #Os métodos Gauss-Jacobi e Gauss-Seidel são métodos iterativos, portanto, devem utilizar a precisão
  #informada no arquivo e para o valor de X0 o valor de G no sistema iterativo Xk= CXk−1 + G.
"""def jacobi(A,precision):
    #128
    n = A.shape[0]
    for k in range(n):"""

def jacobi(A,B,precision,tol,N):  

    dimensionM = A.shape[0]  
    MI = np.zeros(dimensionM)  
    it = 0  
    #iteracoes 
    while (k < N):  
        it = it+1  
        #iteracao de Jacobi  
        for i in np.arange(dimensionM):  
            MI[i] = B[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,dimensionM))):  
                MI[i] -= A[i,j]*precision[j]  
            MI[i] /= A[i,i]
        #tolerancia  
        if (np.linalg.norm(MI-precision,np.inf) < tol):  
            return MI  
        #prepara nova iteracao 
        precision = np.copy(MI)  
    raise NameError('num. max. de iteracoes excedido.')

def jacobiA(A,B,N):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    dimensionM = A.shape[0]  
    MI = np.zeros(dimensionM)

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        MI = (B - np.dot(R,MI)) / D
    return MI

def saveOutput(output, X):
	np.savetxt(output, X, delimiter=',', header='Resposta')

if __name__ == '__main__':
    inputs, outputs = readArgs()
    A, B, precision, simet = readFile(inputs)

    N = 25

    L = jacobiA(A,B,N)
    #saveOutput(outputs, A)
    #C, X = np.array_split(B,2)
    #print(A)
    #print(gauss(A, B))
    #L = cholesky(A)
    print(L)
