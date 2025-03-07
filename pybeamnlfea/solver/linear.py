import numpy as np 
from scipy.sparse.linalg import spsolve
from abc import ABC, abstractmethod

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Solver(ABC):
    @abstractmethod
    def solve(self, assembler):
        pass

class LinearSolver(Solver):
    def __init__(self, solver_type='direct'):
        """
        Linear solver class.
        
        Args:
            solver_type: Type of solver to use ('direct' or 'iterative')
        """
        self.solver_type = solver_type
        
    def solve(self, assembler):
        """
        Solve the linear system K·u = F.
        """
        # Assemble system matrices
        K = assembler.assemble_stiffness_matrix()
        F = assembler.assemble_force_vector()
        
        # Solve the system using sparse solver
        if self.solver_type == 'direct':
            # Direct solver for Ku = F
            u = spsolve(K, F)

        elif self.solver_type == 'iterative':
            # Iterative solvers for large systems
            from scipy.sparse.linalg import cg
            u, info = cg(K, F)
            if info != 0:
                print(f"Warning: Conjugate gradient did not converge. Info: {info}")
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
        # Calculate nodal displacements forces 
        nodal_displacements = self._get_nodal_displacements(assembler, u)

        return nodal_displacements
    
    def _get_nodal_displacements(self, assembler, u):
        """
        Create a dictionary mapping node IDs and DOFs to displacements.
        """
        nodal_displacements = {}
        for (node_id, dof_idx), global_dof in assembler.dof_map.items():
            if global_dof >= 0: # Unconstrained DOFs
                nodal_displacements[(node_id, dof_idx)] = u[global_dof]
            else:
                # For constrained DOFs, displacement is zero
                nodal_displacements[(node_id, dof_idx)] = 0.0

        return nodal_displacements
    