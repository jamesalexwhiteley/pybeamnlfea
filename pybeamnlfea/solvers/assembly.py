import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Assembler:
    def __init__(self, elements, nodes):
        self.elements = elements
        self.nodes = nodes

    def __init__(self, frame):
        self.frame = frame
    
    def assemble_stiffness_matrix(self):
        pass
        
    def assemble_force_vector(self):
        pass
        