import sys
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from os import listdir

def readFile(inputs):
    shape = tuple(np.loadtxt(fname=inputs, dtype=int, delimiter=',', max_rows=1, usecols=(0,1)))
    PreSim = np.loadtxt(fname=inputs, dtype=str, delimiter=',', max_rows=1, usecols=(2,3))
    A = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=',', skiprows=1, max_rows=shape[1], usecols=np.arange(0,shape[1]))
    B = np.loadtxt(fname=inputs, dtype=np.float64, delimiter=',', skiprows=shape[1], max_rows=shape[0], usecols=np.arange(0,shape[1]))

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
    (dimensionM, dimensionN) = A.shape
    for i in range(1, dimensionM):
        for j in range(i, dimensionM):
            matAUX = 0
            matAUX = A[j][i-1] / A[i-1][i-1]
            for k in range(j,j+1):
                A[k] = A[k] - matAUX * A[i-1]
                B[k] = B[k] - matAUX * B[i-1]

    end = timer()
    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução Gauss: {timing}\n")
    return A

def fatoraLU(A):
    start = timer()
    U = np.copy(A)
    n = np.shape(U)[0]
    L = np.eye(n)
    for j in np.arange(n-1):
        for i in np.arange(j+1,n):
            L[i,j] = U[i,j]/U[j,j]
            for k in np.arange(j+1,n):
                U[i,k] = U[i,k] - L[i,j]*U[j,k]
            U[i,j] = 0

    end = timer()
    timing = timedelta(seconds=end-start)
    print(f"\ntempo de execução LU: {timing}\n")
    return L, U

def saveOutput(output, X):
	np.savetxt(output, X, delimiter=',', header='Resposta')

if __name__ == '__main__':
    inputs, outputs = readArgs()
    A, B, precision, simet = readFile(inputs)
    saveOutput(outputs, A)
    
    """
    print(gauss(A, B, dimensionM))

    L, U = fatoraLU(A)
    print(L)
    print()
    print(U)"""