import numpy as np 
from pybeamnlfea.solver.linear import Solver 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class NewtonRaphsonSolver(Solver):
    def __init__(self, assembler, tol=1e-6, max_iter=100):
        self.assembler = assembler
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, assembler):
        pass

    # Current forces in elements based on total displacement (not incremental) F(q)
    # Tangent stiffness at the current state Ktot
    