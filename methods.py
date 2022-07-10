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
    if len(sys.argv) > 1:
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
    return arq

def gauss(A, B):
    dimensionM = A.shape[0]

    for i in np.arange(1, dimensionM):
        for j in np.arange(i, dimensionM):
            matAUX = A[j][i-1] / A[i-1][i-1]
            for k in np.arange(j,j+1):
                A[k] = A[k] - matAUX * A[i-1]
                B[k] = B[k] - matAUX * B[i-1]

    for k in np.arange(dimensionM):
        for i in np.arange(k-1):

for(K = 0; K < n-1; K++){ // cada etapa
        for(i = K+1; i < n;i++){ // cada linha
            M = A[i][K]/A[K][K]; // Mik = Aik/Akk
            for(j=K; j < n; j++) A[i][j] = A[i][j] - M*A[K][j]; // Li <- Li - Mik * Lk
            B[i] = B[i] - M*B[K];
        }
    }
    for(i = n-1; i >= 0; i--){
        for(j=i+1; j<n;j++) B[i] = B[i]- X[j]*A[i][j];
        X[i] = B[i]/A[i][i];
    }
    return X;

def Gaus(A,B):
    aux = np.copy(E[1,:])
    E[1,:] = np.copy(E[0,:])
    E[0,:] = np.copy(aux)
        print(E)



    return x

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
    timedelta()
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
    MI = np.diag(MI)
    return MI

def jacobiX(A,B,precision):
    start = timer()
    dimensionM = A.shape[0]
    x = np.zeros(dimensionM)

    DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    x0 = DiagA/B
    x0 = np.diag(x0)
    x0 = x0.astype(np.double)
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
            end = timer()
            timing = timedelta(seconds=end-start)
            print(f"\nTempo de execução [Jacobi]: {timing}\n")
            return x
        x0 = np.copy(x)

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


def seidel(A,B,precision):  
    start = timer()
    A = A.astype(np.double)
    B = B.astype(np.double)

    dimensionM = A.shape[0]
    DiagA = np.diagflat(np.diag(A))
    C = A - np.diagflat(np.diag(A))
    x0 = DiagA/B
    x0 = np.diag(x0)
    x0 = x0.astype(np.double)
    x = np.copy(x0)

    D = precision + 1
    while (D > precision):  
        for i in np.arange(dimensionM):  
            x[i] = B[i]
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,dimensionM))):
                x[i] -= A[i,j]*x[j]
            x[i] /= A[i,i]
        d = np.linalg.norm(x-x0,np.inf)
        D = d/max(np.fabs(x))
        print(D)
        if (D < precision):
            end = timer()
            timing = timedelta(seconds=end-start)
            print(f"\ntempo de execução [Seidel]: {timing}\n")
            return x
        x0 = np.copy(x)


if __name__ == '__main__':
    inputs= readArgs()
    A, B, precision, simet = readFile(inputs)


    print("----------------------###-------------------###-------------------")
    L, U = fatoraLU(A) 
    print("----------------------###-------------------###-------------------")  
    #print(L)
    print("----------------------###-------------------###-------------------")
    #print(U)
    print("----------------------###-------------------###-------------------")
    print(cholesky(A))
    print("----------------------###-------------------###-------------------")
    print(jacobi(A,B,precision))
    print("----------------------###-------------------###-------------------")
    print(seidel(A,B,precision))   
    print("----------------------###-------------------###-------------------")
    print(Gaus(A,B)) 


    #print(A)
    #print(B)
    #for i in range(3):
    #print(np.max(np.fabs(X[i]-X2[i])))

    #print(np.linalg.norm(B-A,np.inf))
