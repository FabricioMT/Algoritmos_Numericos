import numpy as np
import sys
from os import listdir

def readFile(path):
    shape = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', max_rows=1, usecols=(0,1,2)))
    shapeStr = np.loadtxt(fname=path, dtype=np.str_, delimiter=',', max_rows=1, usecols=3)
    shapeA = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=1, max_rows=3, usecols=(0,1,2)))   
    shapeB = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=4, max_rows=5, usecols=(0,1,2)))
    A = np.array(shapeA)
    B = np.array(shapeB)
    C = np.array(shape)

    qtdSis = np.float64(shape[0])
    N = np.float64(shape[1])
    P = np.float64(shape[2])
    S = shapeStr
  

    print(C)
    print(A,B)

    return A, B, C

def readArgs():
    if len(sys.argv) > 2:
        file_input = listdir('./inputs/')
        for file in file_input:
            if file == sys.argv[1]:
                arq = 'inputs/' + sys.argv[1]
            
        args = sys.argv[1:]
        print(f"Arguments count: {len(sys.argv)}")
        print(f"Arguments of the script : {args}")   
    else:
        print("Entrada de Dados Inválida !")
        exit(1)
    return arq

def readData():
    readFile(readArgs())

if __name__ == '__main__':
    readData()