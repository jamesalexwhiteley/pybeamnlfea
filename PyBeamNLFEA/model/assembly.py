import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Assembler:
    def __init__(self, elements, nodes):
        self.elements = elements
        self.nodes = nodes

    def global_dof_indices(element):
        # Map local DOFs to global ones
        indices = []
        for node in element.nodes:
            indices.extend([node.id * 6 + i for i in range(6)])  # 6 DOFs per node
        return indices

    def assemble_global_stiffness_matrix(self, elements, num_dofs):
        # Assemble global stiffness matrix 
        K_global = lil_matrix((num_dofs, num_dofs))
        
        for elem in elements:
            K_local = elem.local_stiffness_matrix()
            dofs = self.global_dof_indices(elem)
            
            # Add to global matrix
            for i, i_glob in enumerate(dofs):
                for j, j_glob in enumerate(dofs):
                    K_global[i_glob, j_glob] += K_local[i, j]
                
        return K_global.tocsr()  # Convert to CSR for efficient solving
        
    def global_force_vector(self, applied_loads):
        # Assemble global force vector
        pass