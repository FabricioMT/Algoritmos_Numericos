import numpy as np
from readArqs import Args
from timeit import default_timer as timer
from datetime import timedelta

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


if __name__ == '__main__':
    A, B, precision, simet = Args()

    """
    print(gauss(A, B, dimensionM))

    L, U = fatoraLU(A)
    print(L)
    print()
    print(U)"""