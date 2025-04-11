import numpy as np 
from scipy.sparse.linalg import spsolve, eigsh, eigs
from scipy.linalg import inv
from scipy.sparse import eye, csc_matrix
from pybeamnlfea.solver.linear import LinearSolver 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class EigenSolver(LinearSolver):
    def __init__(self, num_modes=5, solver_type='direct'):
        super().__init__(solver_type=solver_type)
        self.num_modes = num_modes

    def eigen_solve(self, assembler):
        """Solve linear eigenvalue problem."""
        # Linear analysis -> internal member forces 
        nodal_displacements, _ = self.solve(assembler)
        element_internal_forces = self._calculate_element_internal_forces(assembler, nodal_displacements) 
        
        # Classical approach (generalised eigenvalue problem |Km − PKg| = 0)
        try: 
            Ktot = assembler.assemble_stiffness_matrix(geometric_stiffness=True, element_internal_forces=element_internal_forces) 
            Km = assembler.assemble_stiffness_matrix(geometric_stiffness=False)
            Kg = Ktot - Km

            # |Km − PKg| = 0 can be rearranged to |A − (1/λ)I| = 0 where A = Km^-1 @ Kg 
            A = spsolve(Km.tocsc(), (Kg + 1e-10 * eye(Kg.shape[0])).tocsc())
            eigenvalues, eigenvectors = eigs(csc_matrix(A), k=self.num_modes, which='LM')
            eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)

            # Sort results 
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            critical_loads = 1 / eigenvalues

        except Exception as e: 
            print(f"EigenSolver failed with error message: {e}")
            return [], [] 
        
        # Eigenvectors -> nodal displacements (modes)
        buckling_modes = []
        for i in range(self.num_modes):
            mode_displacements = self._get_nodal_displacements(assembler, eigenvectors[:, i])
            buckling_modes.append(mode_displacements)
            
        return critical_loads, buckling_modes
    
    def _calculate_element_internal_forces(self, assembler, nodal_displacements):                
        """Calculate internal forces for each element based on global displacements."""
        element_forces = {}
        
        for element_id, element in assembler.frame.elements.items():
            # Element nodes 
            node_i_id = element.nodes[0].id
            node_j_id = element.nodes[1].id
            
            # Extract nodal displacements
            ndof_i = assembler.frame.nodes[node_i_id].ndof
            ndof_j = assembler.frame.nodes[node_j_id].ndof
            
            u_i = np.array([nodal_displacements.get((node_i_id, dof), 0.0) for dof in range(ndof_i)])
            u_j = np.array([nodal_displacements.get((node_j_id, dof), 0.0) for dof in range(ndof_j)])
            
            # Combine into element displacement vector (in global coordinates)
            u_elem_global = np.concatenate([u_i, u_j])
            
            # Transform to local coordinates
            T = element.compute_transformation_matrix()                                                    
            u_elem_local = T @ u_elem_global
            
            # Calculate internal forces
            internal_forces = element.compute_internal_forces(u_elem_local)
            element_forces[element_id] = internal_forces
            
        return element_forces