import numpy as np
import sys
from os import listdir

def readFile(path):
    shape = tuple(np.loadtxt(fname=path, dtype=np.str_, delimiter=',', max_rows=1, usecols=(0,1,2,3)))
    shapeA = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=1, max_rows=3, usecols=(0,1,2)))   
    shapeB = tuple(np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=4, max_rows=5, usecols=(0,1,2)))
    A = np.array(shapeA)
    B = np.array(shapeB)
    I = np.array(shape)
    return A, B, I

def readArgs():
    if len(sys.argv) > 2:
        arq = 0;
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
    return readFile(arq)

 