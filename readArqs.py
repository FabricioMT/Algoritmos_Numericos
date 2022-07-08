import numpy as np
import sys
from os import listdir

def readFile(path):
    shape = tuple(np.loadtxt(fname=path, dtype=int, delimiter=',', max_rows=1, usecols=(0,1)))
    PreSim = np.loadtxt(fname=path, dtype=str, delimiter=',', max_rows=1, usecols=(2,3))
    A = np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=1, max_rows=shape[1], usecols=np.arange(0,shape[1]))
    B = np.loadtxt(fname=path, dtype=np.float64, delimiter=',', skiprows=shape[1], max_rows=shape[0], usecols=np.arange(0,shape[1]))

    if shape[0] != 1:
        B = np.reshape(B, (shape[0],shape[1],1))
    else:
        B = np.reshape(B, (shape[1],shape[0]))

    A = np.reshape(A, (shape[1],shape[1]))
    P = np.float64(PreSim[0])
    S = PreSim[1]

    return A, B, P, S


def Args():
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
    return readFile(arq)   

 