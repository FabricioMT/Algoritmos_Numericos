from lib2to3.pytree import convert
from sqlite3 import converters
import numpy as np
import sys

from pyparsing import str_type

def readArgs():
    if len(sys.argv) > 2:
        args = sys.argv[1:]
        arq = sys.argv[1]
        print(f"Arguments count: {len(sys.argv)}")
        print(f"Arguments of the script : {args}")    
    else:
        print("Entrada de Dados Inválida !")
        exit(1)
    return arq

def readFile(path):
    shape = tuple(np.loadtxt(fname=path, dtype=str_type, delimiter=',', max_rows=1, usecols=(0,1,2,3)))
    shapeA = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=1, max_rows=3, usecols=(0,1,2)))
    shapeB = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=4, max_rows=5, usecols=(0,1,2)))
    A = np.array(shapeA)
    B = np.array(shapeB)

    qtdSis = np.float64(shape[0])
    N = np.float64(shape[1])
    P = np.float64(shape[2])
    S = shape[3].strip()

    print(qtdSis,N,P,S)
    print(A,B)
    #print(np.array(shapeA))
    

if __name__ == "__main__":
    readFile(readArgs())
