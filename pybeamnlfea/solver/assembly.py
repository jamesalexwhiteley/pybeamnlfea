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
                has_finite_stiffness = False
                
                if node_id in self.frame.boundary_conditions:
                    bc = self.frame.boundary_conditions[node_id]
                    if bc.is_dof_constrained(i):
                        if bc.has_finite_stiffness(i):
                            has_finite_stiffness = True
                        else:
                            is_constrained = True
                
                if not is_constrained:
                    # Both unconstrained DOFs and elastically supported DOFs get a DOF index
                    self.dof_map[(node_id, i)] = current_dof
                    current_dof += 1
                else:
                    # Assign negative index to indicate fully constrained DOF
                    self.dof_map[(node_id, i)] = -1
        
        self.total_dofs = current_dof

    def assemble_stiffness_matrix(self, include_constrained_dofs=False, geometric_stiffness=False, element_internal_forces=None):
        """
        Assemble the global stiffness matrix.
        
        Args:
            include_constrained_dofs: If True, includes constrained DOFs in the matrix
            geometric_stiffness: If True, includes geometric stiffness effects
            element_internal_forces: Dictionary mapping element IDs -> internal forces (required if include_geometric=True)
        
        Returns:
            K: Global stiffness matrix (sparse CSR format)
        """
        if geometric_stiffness and element_internal_forces is None:
            raise ValueError("Element internal forces must be provided when including geometric effects")
        
        # Sparse matrix in COO format 
        rows, cols, data = [], [], []
    
        if include_constrained_dofs:
            total_dofs = sum(node.ndof for node in self.frame.nodes.values())
        else:
            total_dofs = self.total_dofs
        
        # Assemble element contributions
        for element_id, element in self.frame.elements.items():
            # Get element stiffness matrix 
            if geometric_stiffness:
                internal_forces = element_internal_forces.get(element_id, {})
                k_local = element.compute_local_stiffness_matrix(include_geometric=True, internal_forces=internal_forces)
            else:
                k_local = element.compute_local_stiffness_matrix(include_geometric=False)

            # Transform to global coordinates 
            T = element.compute_transformation_matrix()
            k_elem = T.transpose() @ k_local @ T
            
            # Get element global DOF indices 
            element_dofs = []
            
            if include_constrained_dofs:
                # Use full DOF numbering including constrained DOFs
                for node in element.nodes:
                    base_index = node.id * node.ndof 
                    for i in range(node.ndof):
                        element_dofs.append(base_index + i)
            else:
                # Use mapping that excludes constrained DOFs
                for node in element.nodes:
                    for i in range(node.ndof):
                        global_dof = self.dof_map.get((node.id, i), -1)
                        element_dofs.append(global_dof)

            # Iterate through all DOF pairs 
            for i_local in range(len(element_dofs)):
                i_global = element_dofs[i_local]
                
                # Skip constrained DOFs if necessary 
                if not include_constrained_dofs and i_global < 0:
                    continue
                    
                for j_local in range(len(element_dofs)):
                    j_global = element_dofs[j_local]
                    
                    # Skip constrained DOFs if necessary 
                    if not include_constrained_dofs and j_global < 0:
                        continue
                        
                    rows.append(i_global)
                    cols.append(j_global)
                    data.append(k_elem[i_local, j_local])
        
        # Add contributions from elastic supports (finite stiffness)
        for node_id, bc in self.frame.boundary_conditions.items():
            node = self.frame.nodes[node_id]
            
            for i in range(node.ndof):
                # Check if this DOF has a finite stiffness support
                if bc.has_finite_stiffness(i):
                    global_dof = self.dof_map.get((node_id, i), -1)
                    
                    # Skip if this DOF is not in our system (should not happen with finite stiffness)
                    if global_dof < 0:
                        continue
                    
                    # Add the support stiffness to the diagonal of K
                    support_stiffness = bc.get_support_stiffness(i)
                    rows.append(global_dof)
                    cols.append(global_dof)
                    data.append(support_stiffness)
        
        K = coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs))
        return K.tocsr()
    
    def assemble_force_vector(self):
        """Assemble the global force vector for the unconstrained system."""
        # Initialise force vector with zeros for all DOFs
        F = np.zeros(self.total_dofs)
        
        # Add nodal loads to the global force vector
        for load in self.frame.loads.values():
            if isinstance(load, NodalLoad):
                node_id = load.node_id
                force_vector = load.force_vector
                
                # Map each component of the force vector to the global DOF
                for local_dof in range(self.frame.nodes[node_id].ndof): # [Fx, Fy, Fz, Mx, My, Mz, Bx]
                    global_dof = self.dof_map.get((node_id, local_dof), -1)
                    
                    # Only add force if DOF is not constrained
                    if global_dof >= 0:
                        F[global_dof] += force_vector[local_dof]
        
        # Add support settlement/displacement effects for finite stiffness supports
        for node_id, bc in self.frame.boundary_conditions.items():
            node = self.frame.nodes[node_id]
            
            for i in range(node.ndof):
                # Check if this DOF has a finite stiffness support with prescribed displacement
                if bc.has_finite_stiffness(i) and bc.has_prescribed_displacement(i):
                    global_dof = self.dof_map.get((node_id, i), -1)
                    
                    # Skip if this DOF is not in our system
                    if global_dof < 0:
                        continue
                    
                    # Calculate force from support displacement: F = k * u
                    support_stiffness = bc.get_support_stiffness(i)
                    prescribed_displacement = bc.get_prescribed_displacement(i)
                    F[global_dof] += support_stiffness * prescribed_displacement
        
        return F