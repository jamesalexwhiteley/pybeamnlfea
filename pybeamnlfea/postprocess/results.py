import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Results:
    def __init__(self, assembler, global_displacements, element_forces=None):
        """
        Results class to store and process solution results.
        
        Parameters
        ----------
            assembler: The assembler used to build the system
            global_displacements: Dictionary mapping (node_id, dof_idx) to displacement values in global coords
            element_forces: Dictionary of element forces and local displacements
        """
        self.assembler = assembler
        self.frame = assembler.frame
        self.global_displacements = global_displacements
        self.element_forces = element_forces
        
        # Calculate local displacements for each element
        self.local_element_displacements = self._calculate_local_displacements()
    
    def _calculate_local_displacements(self):
        """
        Transform global displacements to local coordinate system for each element.
        """
        local_displacements = {}
        
        for element_id, element in self.frame.elements.items():
            # Get transformation matrix for this element
            T = element.compute_transformation_matrix()
            
            # Get global displacements for all DOFs of this element
            global_disp_vector = []
            for node in element.nodes:
                for i in range(node.ndof):
                    disp = self.global_displacements.get((node.id, i), 0.0)
                    global_disp_vector.append(disp)
                    
            global_disp_vector = np.array(global_disp_vector)
            
            # Transform to local coordinates
            # Note: T transforms local->global, so T.transpose() transforms global->local
            local_disp_vector = T.transpose() @ global_disp_vector
            
            # Store by element ID
            local_displacements[element_id] = local_disp_vector
            
        return local_displacements
    
    def get_nodal_displacement(self, node_id, dof_idx=None, coordinate_system='global'):
        """
        Get displacement for a specific node.
        """
        if coordinate_system == 'global':
            if dof_idx is not None:
                return self.global_displacements.get((node_id, dof_idx), 0.0)
            else:
                # Return all DOFs for this node
                node_disps = {}
                for k, v in self.global_displacements.items():
                    if k[0] == node_id:
                        node_disps[k[1]] = v
                return node_disps
        else:
            raise ValueError("For local displacements, use get_element_local_displacement method")
    
    def get_element_local_displacement(self, element_id):
        """
        Get displacements for an element in its local coordinate system.
        """
        return self.local_element_displacements.get(element_id, None)
    
    def get_element_forces(self, element_id):
        """
        Get forces for a specific element (already in local coordinates).
        """
        return self.element_forces.get(element_id, None)
    
    def calculate_element_deflection(self, element_id, num_points=100):
        """
        Calculate deflected shape of an element in its local coordinate system
        """
        element = self.frame.elements[element_id]
        L = element.length
        local_disps = self.local_element_displacements[element_id]
        
        # Create points along element length
        c = np.linspace(0, L, num_points)
        deflection = np.zeros((num_points, 6))  # For 6 deflection components (w, u, v, θz, θx, θy)
        
        if len(element.nodes) == 2 and len(local_disps) == 14:
            # First node 
            w1, u1, v1 = local_disps[0], local_disps[1], local_disps[2]  # Translations 
            theta_z1, theta_x1, theta_y1, phi1 = local_disps[3], local_disps[4], local_disps[5], local_disps[6]  # Rotations
            
            # Second node 
            w2, u2, v2 = local_disps[7], local_disps[8], local_disps[9]  # Translations 
            theta_z2, theta_x2, theta_y2, phi2 = local_disps[10], local_disps[11], local_disps[12], local_disps[13]  # Rotations
            
            for i, z in enumerate(c):
                # Calculate all 14 shape functions at point z
                N = [
                    (L - z)/L,                          # N1
                    (L**3 - 3*L*z**2 + 2*z**3)/L**3,    # N2
                    (L**3 - 3*L*z**2 + 2*z**3)/L**3,    # N3
                    (L**3 - 3*L*z**2 + 2*z**3)/L**3,    # N4
                    -z + 2*z**2/L - z**3/L**2,          # N5
                    z - 2*z**2/L + z**3/L**2,           # N6
                    z - 2*z**2/L + z**3/L**2,           # N7
                    z/L,                                # N8
                    z**2*(3*L - 2*z)/L**3,              # N9
                    z**2*(3*L - 2*z)/L**3,              # N10
                    z**2*(3*L - 2*z)/L**3,              # N11
                    z**2*(L - z)/L**2,                  # N12
                    z**2*(-L + z)/L**2,                 # N13
                    z**2*(-L + z)/L**2                  # N14
                ]

                u_prime1, u_prime2 = theta_y1, theta_y2
                v_prime1, v_prime2 = theta_x1, theta_x2
                # theta1, theta2, theta_prime1, theta_prime2 = theta_z1, theta_z2, phi1, phi2
                
                # Axial deflection (w)
                deflection[i, 0] = N[0]*w1 + N[7]*w2

                # Transverse deflection in x-direction (ū)
                deflection[i, 1] = N[1]*u1 + N[5]*u_prime1 + N[8]*u2 + N[12]*u_prime2

                # Transverse deflection in y-direction (v̄)
                deflection[i, 2] = N[2]*v1 + N[4]*v_prime1 + N[9]*v2 + N[11]*v_prime2
        
        return c, deflection
