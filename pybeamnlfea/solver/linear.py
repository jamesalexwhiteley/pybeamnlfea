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
        Solve the linear system Ku = F 
        """
        # Assemble system matrices (unconstrained DOFs)
        K = assembler.assemble_stiffness_matrix()
        F = assembler.assemble_force_vector()
        
        # Solve the system using sparse solver
        if self.solver_type == 'direct':
            # Direct solver for Ku = F
            u = spsolve(K, F) # Unconstrained DOFs

        elif self.solver_type == 'iterative':
            # Iterative solvers for large systems
            from scipy.sparse.linalg import cg
            u, info = cg(K, F) # Unconstrained DOFs
            if info != 0:
                print(f"Warning: Conjugate gradient did not converge. Info: {info}")
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
        # Calculate nodal displacements and reaction forces 
        nodal_displacements = self._get_nodal_displacements(assembler, u)
        nodal_forces = self.solve_reaction_forces(assembler, nodal_displacements)

        return nodal_displacements, nodal_forces 
    
    def solve_reaction_forces(self, assembler, nodal_displacements):
        """Calculate reaction forces via F_all = K_full @ u_full"""
        K_full = assembler.assemble_full_stiffness_matrix()
        total_dofs = sum(node.ndof for node in assembler.frame.nodes.values())
        u_full = np.zeros(total_dofs)
        
        # Fill displacement values 
        for (node_id, dof_idx), disp in nodal_displacements.items():
            global_dof = node_id * assembler.frame.nodes[node_id].ndof + dof_idx
            u_full[global_dof] = disp
        
        # Forward problem 
        F_all = K_full @ u_full 
        
        # Extract forces 
        nodal_forces = {}
        for node_id, node in assembler.frame.nodes.items():
            for dof_idx in range(node.ndof):
                global_dof = node_id * node.ndof + dof_idx
                nodal_forces[(node_id, dof_idx)] = F_all[global_dof]

        return nodal_forces

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

