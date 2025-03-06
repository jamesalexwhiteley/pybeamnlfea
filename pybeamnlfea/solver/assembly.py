import numpy as np 
from pybeamnlfea.model.load import NodalLoad 
from scipy.sparse import coo_matrix, csr_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Assembler:
    def __init__(self, frame):
        self.frame = frame
        self.dof_map = {}  # Maps (node_id, dof_index) to global DOF index
        self._build_dof_map()
        
    def _build_dof_map(self):
        """Create mapping between node DOFs and global matrix indices."""
        current_dof = 0
        
        for node_id, node in self.frame.nodes.items():
            dofs_per_node = node.ndof 
            
            for i in range(dofs_per_node):
                # Check if this DOF is constrained by a boundary condition
                is_constrained = False
                if node_id in self.frame.boundary_conditions:
                    bc = self.frame.boundary_conditions[node_id]
                    if bc.is_dof_constrained(i):  
                        is_constrained = True
                
                if not is_constrained:
                    self.dof_map[(node_id, i)] = current_dof
                    current_dof += 1
                else:
                    # Assign negative index to indicate constrained DOF
                    self.dof_map[(node_id, i)] = -1
        
        self.total_dofs = current_dof

    # def assemble_stiffness_matrix(self):
    #     """Assemble the global stiffness matrix using a direct approach."""
    #     # Initialize sparse matrix in COO format 
    #     rows, cols, data = [], [], []
        
    #     # Assemble contributions from each element
    #     for element_id, element in self.frame.elements.items():
    #         # Get element stiffness matrix
    #         k_local = element.compute_local_stiffness_matrix() 

    #         # Transform to global coordinates 
    #         T = element.compute_transformation_matrix()
    #         k_elem = T.transpose() @ k_local @ T
            
    #         # Get global DOF indices for this element 
    #         element_dofs = []
    #         for node in element.nodes:
    #             for i in range(node.ndof):
    #                 global_dof = self.dof_map.get((node.id, i), -1)
    #                 element_dofs.append(global_dof)

    #         # Simpler approach: Iterate through all DOF pairs and add non-constrained ones
    #         for i_local in range(len(element_dofs)):
    #             i_global = element_dofs[i_local]
    #             if i_global < 0:  # Skip constrained DOFs
    #                 continue
                    
    #             for j_local in range(len(element_dofs)):
    #                 j_global = element_dofs[j_local]
    #                 if j_global < 0:  # Skip constrained DOFs
    #                     continue
                        
    #                 # Add to sparse matrix components
    #                 rows.append(i_global)
    #                 cols.append(j_global)
    #                 data.append(k_elem[i_local, j_local])
        
    #     # Create the sparse matrix in COO format and convert to CSR
    #     K = coo_matrix((data, (rows, cols)), shape=(self.total_dofs, self.total_dofs))
    #     return K.tocsr()

    def assemble_stiffness_matrix(self):
        """Assemble the global stiffness matrix."""
        # Initialize the global stiffness matrix with zeros
        K = np.zeros((self.total_dofs, self.total_dofs))
        
        # Assemble contributions from each element
        for element_id, element in self.frame.elements.items():
            # Get element stiffness matrix in local coordinates
            k_local = element.compute_local_stiffness_matrix()
            
            # Transform to global coordinates
            T = element.compute_transformation_matrix()
            k_elem = T.transpose() @ k_local @ T                                    
            
            # Get global DOF indices for this element
            element_dofs = []
            for node in element.nodes:
                for i in range(node.ndof):
                    global_dof = self.dof_map.get((node.id, i), -1)
                    element_dofs.append(global_dof)
            
            # Add element stiffness contributions to the global stiffness matrix
            for i_local, i_global in enumerate(element_dofs):
                if i_global < 0:  # Skip constrained DOFs
                    continue
                    
                for j_local, j_global in enumerate(element_dofs):
                    if j_global < 0:  # Skip constrained DOFs
                        continue
                    
                    # Add element contribution to global stiffness matrix
                    K[i_global, j_global] += k_elem[i_local, j_local]
        
        return csr_matrix(K)
    
    def assemble_force_vector(self):
        """
        Assemble the global force vector from loads.
        """
        # Initialize force vector with zeros for all DOFs
        F = np.zeros(self.total_dofs)
        
        # Add nodal loads to the global force vector
        for load in self.frame.loads.values():
            if isinstance(load, NodalLoad):
                node_id = load.node_id
                force_vector = load.force_vector
                
                # Map each component of the force vector to the global DOF
                for local_dof in range(self.frame.nodes[node_id].ndof): # [Fz, Fx, Fy, Mz, Mx, My, Bz]
                    global_dof = self.dof_map.get((node_id, local_dof), -1)
                    
                    # Only add force if DOF is not constrained
                    if global_dof >= 0:
                        F[global_dof] += force_vector[local_dof]
        
        return F
        
