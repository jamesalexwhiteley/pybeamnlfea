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
            self.u = spsolve(K, F)

        elif self.solver_type == 'iterative':
            # Iterative solvers for large systems
            from scipy.sparse.linalg import cg
            u, info = cg(K, F)
            if info != 0:
                print(f"Warning: Conjugate gradient did not converge. Info: {info}")
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
        # Create a dictionary mapping node IDs and DOFs to displacements
        displacements = {}
        for (node_id, dof_idx), global_dof in assembler.dof_map.items():
            if global_dof >= 0: # Unconstrained DOFs
                displacements[(node_id, dof_idx)] = self.u[global_dof]
            else:
                # For constrained DOFs, displacement is zero
                displacements[(node_id, dof_idx)] = 0.0
        
        # Calculate element forces 
        element_forces = self._calculate_element_results(assembler, self.u)
        
        return displacements, element_forces
    
    def _calculate_element_results(self, assembler, u):
        """
        Calculate element forces and stresses.
        """
        frame = assembler.frame
        element_forces = {}
        
        for element_id, element in frame.elements.items():
            # Get element DOFs
            element_dofs = []
            element_u = []
            
            for node in element.nodes:
                for i in range(node.ndof):
                    global_dof = assembler.dof_map.get((node.id, i), -1)
                    element_dofs.append(global_dof)
                    
                    # Get displacement value (0 if constrained)
                    if global_dof >= 0:
                        element_u.append(u[global_dof])
                    else:
                        element_u.append(0.0)
            
            # Convert to numpy array
            element_u = np.array(element_u)
            
            # Get transformation matrix
            T = element.compute_transformation_matrix()
            
            # Transform displacements to local coordinates
            local_u = T @ element_u
            
            # Compute local element forces
            k_local = element.compute_local_stiffness_matrix()
            local_forces = k_local @ local_u
            
            # Store results
            element_forces[element_id] = {
                'local_displacements': local_u,
                'local_forces': local_forces,
                # 'global_displacements': element_u
            }

        return element_forces