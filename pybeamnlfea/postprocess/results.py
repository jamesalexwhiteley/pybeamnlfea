import numpy as np 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Results:
    def __init__(self, assembler, displacements, element_forces=None):
        """
        Results class to store and process solution results.
        
        Args:
            assembler: The assembler used to build the system
            displacements: Dictionary mapping (node_id, dof_idx) to displacement values
            element_forces: Dictionary of element forces and local displacements
        """
        self.assembler = assembler
        self.displacements = displacements
        self.element_forces = element_forces
        self.frame = assembler.frame
        
    def get_nodal_displacement(self, node_id, dof_idx=None):
        """
        Get displacement for a specific node.
        """
        if dof_idx is not None:
            return self.displacements.get((node_id, dof_idx), 0.0)
        else:
            # Return all DOFs for this node
            node_disps = {}
            for k, v in self.displacements.items():
                if k[0] == node_id:
                    node_disps[k[1]] = v
            return node_disps
    
    def get_element_forces(self, element_id):
        """
        Get forces for a specific element.
        """
        return self.element_forces.get(element_id, None)
    
    def get_max_displacement(self):
        """
        Get the maximum displacement magnitude and its location.
        """
        max_disp = 0.0
        max_node = None
        max_dof = None
        
        for (node_id, dof_idx), disp in self.displacements.items():
            if abs(disp) > max_disp:
                max_disp = abs(disp)
                max_node = node_id
                max_dof = dof_idx
                
        return max_disp, max_node, max_dof
    
    # def calculate_reactions(self):
    #     """
    #     Calculate reaction forces at constrained nodes.
    #     """
    #     # Get system matrices
    #     K = self.assembler.assemble_stiffness_matrix()
        
    #     # Create full displacement vector (with zeros at constrained DOFs)
    #     n_dofs = self.assembler.total_dofs
    #     u_full = np.zeros(n_dofs)
        
    #     for (node_id, dof_idx), disp in self.displacements.items():
    #         global_dof = self.assembler.dof_map.get((node_id, dof_idx), -1)
    #         if global_dof >= 0:
    #             u_full[global_dof] = disp
        
    #     # Calculate global forces F = K·u
    #     F_global = K @ u_full
        
    #     # Extract reactions at constrained DOFs
    #     reactions = {}
    #     for node in self.frame.nodes.values():
    #         for i, constrained in enumerate(node.constraints):
    #             if constrained:
    #                 # This DOF is constrained, get the reaction force
    #                 global_dof = self.assembler.get_global_dof(node.id, i, include_constrained=True)
    #                 reactions[(node.id, i)] = F_global[global_dof]
        
    #     return reactions
    