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
    B = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=' ', skiprows=(shape[1]+1), max_rows=shape[0], usecols=np.arange(0,shape[1]))

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


def jacobi(A,B,precision):

    dimensionM = A.shape[0]
    x = np.zeros(dimensionM)

    DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    x0 = DiagA/B
    x0 = np.diag(x0)

    D = precision + 1
    while (D > precision):  
        for i in np.arange(dimensionM):  
            x[i] = B[i]
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,dimensionM))):
                x[i] -= A[i,j]*x0[j]
            x[i] /= A[i,i]

        d = np.linalg.norm(x-x0,np.inf)  
        D = d/max(np.fabs(x))
        print(D)
        if (D < precision):
            return x
        x0 = np.copy(x)

def jacobiA(A,b,precision):                                                                                                                                                          
    dimensionM = A.shape[0]
    x = np.zeros(dimensionM)
    DiagA = np.diagflat(np.diag(A))
    x0 = DiagA/B                                                                                                                                                              
    V = np.diag(A)
    R = A - np.diagflat(V)
    D = precision +1
    while (D > precision):  
        for i in np.arange(dimensionM):
            x = (b - np.dot(R,x)) / V

        d = np.linalg.norm(x-x0,np.inf) 
        D = d/(np.max(np.fabs(x)))

        if (D < precision):
            return x
        x0 = np.copy(x)                                                                                                                                                                        


def saveOutput(output, X):
	np.savetxt(output, X, delimiter=',', header='Resposta')

if __name__ == '__main__':
    inputs, outputs = readArgs()
    A, B, precision, simet = readFile(inputs)

    N = 25

    L = jacobi(A,B,precision)
    print(L)
    #print(gauss(A, B))
    #L = cholesky(A)
    """DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    G = DiagA/B
    print(DiagA)
    print(C)"""

    #print(A)
    #print(B)
    #for i in range(3):
    #print(np.max(np.fabs(X[i]-X2[i])))

    #print(np.linalg.norm(B-A,np.inf))
