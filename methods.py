import numpy as np
from readArqs import readData

def gauss(A,B):
    linhaA = A.shape
    for i in range(1, linhaA):
        for j in range(i, linhaA):
            matAUX = 0
            matAUX = A[j][i-1] / A[i-1][i-1]						
            for k in range(j,j+1):
                A[k] = A[k] - matAUX * A[i-1]
                B[k] = B[k] - matAUX * B[i-1]

def fatoraLU(A):  
    U = np.copy(A)  
    n = np.shape(U)[0]  
    L = np.eye(n)  
    for j in np.arange(n-1):  
        for i in np.arange(j+1,n):  
            L[i,j] = U[i,j]/U[j,j]  
            for k in np.arange(j+1,n):  
                U[i,k] = U[i,k] - L[i,j]*U[j,k]  
            U[i,j] = 0  
    return L, U

if __name__ == '__main__':
    readData()
