import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from abc import ABC, abstractmethod

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Solver(ABC):
    @abstractmethod
    def solve(self, assembler):
        pass

class LinearSolver(Solver):
    def solve(self, assembler):
        pass