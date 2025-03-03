import numpy as np 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Results:
    def __init__(self, model, displacements):
        self.model = model
        self.displacements = displacements
        
    def compute_stresses(self):
        # Calculate stresses at integration points
        pass
        
    def compute_internal_forces(self):
        # Calculate internal member forces
        pass
    