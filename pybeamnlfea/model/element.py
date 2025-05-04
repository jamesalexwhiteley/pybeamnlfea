import numpy as np 
from pybeamnlfea.model.node import Node 
from pybeamnlfea.model.material import Material 
from pybeamnlfea.model.section import Section 
from pybeamnlfea.utils.stiffness_matrix import thin_wall_stiffness_matrix

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Element:
    """
    Base element class. 
    """
    def __init__(self, **kwargs):
        """
        Initialise a generic element.
        """
        # Store additional properties
        for key, value in kwargs.items():
            setattr(self, key, value)

class ThinWalledBeamElement(Element):
    def __init__(self, id: int, nodes: Node, material: Material, section: Section): 
        self.id = id 
        self.material = material
        self.section = section

        # Initial state 
        self.nodes = nodes  
        self._initialise_local_axes() 
        self.initial_state = {
            'coords': [node.coords.copy() for node in nodes],
            'R': self.R.copy(),
            'L': self.L
        }
        
        # Need to track deformation steps as analysis progresses 
        self.current_state = self.initial_state.copy()
        self.previous_state = self.initial_state.copy()

    def __repr__(self) -> str:
        """String representation of the element."""
        return f"Element({self.id}, nodes={self.nodes})"

    def _initialise_local_axes(self):
        """
        Compute local coordinate system
            - local x axis is along the member direction
            - local y and z axes are derived from projections of global y and z
        """
        node0, node1 = self.nodes[0], self.nodes[1]
        self.L = node0.distance_to(node1)

        a = self.nodes[1].coords - self.nodes[0].coords
        x_local = a / np.linalg.norm(a)

        # "reference up" vector
        ref_up = np.array([0, 0, 1], dtype=float)

        # try cross(x_local, ref_up)
        y_l_temp = np.cross(x_local, ref_up)
        nrm = np.linalg.norm(y_l_temp)
        if nrm < 1e-12:
            # else
            ref_up = np.array([0, 1, 0], dtype=float)
            y_l_temp = np.cross(x_local, ref_up)
            nrm = np.linalg.norm(y_l_temp)
            if nrm < 1e-12:
                return np.eye(3)

        y_l_temp /= nrm
        # now z_l = x_l x y_l
        z_local = np.cross(x_local, y_l_temp)
        z_local /= np.linalg.norm(z_local)
        # refine y_l to ensure orthogonality
        y_local = np.cross(z_local, x_local)
        y_local /= np.linalg.norm(y_local)

        self.R = np.vstack([x_local, y_local, z_local])

    def compute_elastic_stiffness_matrix(self): 
        """Compute local elastic stiffness matrix (without geometric effects).""" 
        # Section and material properties 
        A = self.section.A
        Iy = self.section.Iy
        Iz = self.section.Iz
        J = self.section.J
        Iw = self.section.Iw
        
        E = self.material.E
        G = self.material.G
        L = self.L
        
        # Elastic stiffness matrix 
        k_elastic = thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
                                            0, 0, 0, 0,         # forces
                                            0, 0, 0, 0, 0, 0)   # geometry  
        
        return k_elastic

    def compute_local_stiffness_matrix(self, include_geometric=False, internal_forces=None):
        """Compute local stiffness matrix with optional geometric effects."""
        # Section and material properties
        A = self.section.A
        Iy = self.section.Iy
        Iz = self.section.Iz
        J = self.section.J
        Iw = self.section.Iw
        
        E = self.material.E
        G = self.material.G
        L = self.L
        
        # Default force parameters
        P0 = 0
        My0 = 0
        Mz0 = 0
        B0_bar = 0
        
        # Other geometric parameters
        W_bar = 0
        y0 = 0
        z0 = 0
        beta_y = 0
        beta_z = 0
        r = 0
        
        if include_geometric and internal_forces is not None:
            P0 = internal_forces.get('axial', 0)
            My0 = internal_forces.get('moment_y', 0)
            Mz0 = internal_forces.get('moment_z', 0)
            B0_bar = internal_forces.get('bimoment', 0)
        
        # Stiffness matrix 
        k = thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
                                    P0, My0, Mz0, B0_bar, 
                                    W_bar, y0, z0, beta_y, beta_z, r)
        
        return k

    def compute_geometric_stiffness_matrix(self, internal_forces):
        """
        Compute only the geometric stiffness matrix component.
        """
        k_full = self.compute_local_stiffness_matrix(include_geometric=True, internal_forces=internal_forces)
        k_elastic = self.compute_elastic_stiffness_matrix()

        return k_full - k_elastic
        
    def compute_internal_forces(self, local_displacements):
        """
        Compute internal forces from element local displacements in local coordinates.
        """
        # Elastic stiffness matrix 
        k_elastic = self.compute_elastic_stiffness_matrix()
        internal_forces_vector = k_elastic @ local_displacements # F = Ku in local coordinates

        # Map to physical quantities based on the DOF ordering for this element
        # [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Bx1, Fx2, Fy2, Fz2, Mx2, My2, Mz2, Bx2]
    
        # Extract internal forces 
        P1 =  internal_forces_vector[0]   # Axial force at start
        P2 = -internal_forces_vector[7]   # Axial force at end (negative for equilibrium)
        
        My1 = internal_forces_vector[4]   
        My2 = internal_forces_vector[11] 
        
        Mz1 = internal_forces_vector[5]   
        Mz2 = internal_forces_vector[12]  
        
        B1 =  internal_forces_vector[6]   
        B2 =  internal_forces_vector[13]    
        
        # For geometric stiffness, take average quantities 
        P = (P1 + P2) / 2
        My = (My1 + My2) / 2 * (1 if My1 + My2 >= 0 else -1)
        Mz = (Mz1 + Mz2) / 2 * (1 if Mz1 + Mz2 >= 0 else -1)
        B = (B1 + B2) / 2 * (1 if B1 + B2 >= 0 else -1)
        
        return {
            'axial': P,
            'moment_y': My,
            'moment_z': Mz,
            'bimoment': B
        }
    
    def compute_local_to_global_transformation_matrix(self):
        """Compute the transformation matrix for this element."""
        
        # Rotation matrix  
        core_block = np.zeros((7, 7))
        core_block[:3, :3] = self.R
        core_block[3:6, 3:6] = self.R
        core_block[6, 6] = 1
        
        I2 = np.eye(2)
        Q = np.kron(I2, core_block)
        
        return Q
    
    def compute_controid_transformation_matrix(self): 
        """Compute the -> centroidal system transformation matrix for this element."""

        C = np.eye(14)  
        
        # Offset between shear center and centroid
        y0 = self.section.y0  
        z0 = self.section.z0  
        
        # Non-zero off-diagonal terms 
        C[2, 3] = y0 
        C[5, 6] = y0 
        C[9, 10] = y0 
        C[12, 13] = y0  
        
        C[1, 3] = z0 
        C[4, 6] = z0  
        C[8, 10] = z0  
        C[11, 13] = z0  
        
        return C

    def compute_transformation_matrix(self):
        """Compute the transformation matrix T = (CQC^-1)^-1."""
        # C = self.compute_controid_transformation_matrix()
        # Q = self.compute_local_to_global_transformation_matrix()
        # self.T = np.linalg.inv(C @ Q @ np.linalg.inv(C))   
        self.T = self.compute_local_to_global_transformation_matrix()                       # NOTE                  
 
        return self.T
    
    # def update_state(self, displacements):
    #     """
    #     Update the current state (position and local axes) based on element nodal displacements,
    #     using Rodrigues' rotation formula to handle rotations.
        
    #     Args:
    #         displacements: List of displacement vectors [u1, u2] for nodes
    #     """
    #     # 0. Store the previous state 
    #     self.previous_state = self.current_state.copy()

    #     # 1. Update nodal positions
    #     self.current_coords = [
    #         self.nodes[0].coords + displacements[0],
    #         self.nodes[1].coords + displacements[1]
    #     ]
        
    #     # 2. Get new tangent vector 
    #     delta = self.current_coords[1] - self.current_coords[0]
    #     self.current_L = np.linalg.norm(delta)
    #     c_new = delta / self.current_L  
        
    #     # 3. Get previous tangent vector 
    #     c_old = self.R[0, :] # First row is x-axis (tangent)
        
    #     # 4. Calculate rotation axis (omega)
    #     omega = np.cross(c_old, c_new)
    #     omega_norm = np.linalg.norm(omega)
        
    #     # Handle special case - vectors parallel 
    #     if omega_norm < 1e-10:
    #         if np.dot(c_old, c_new) > 0:
    #             # Same direction
    #             self.current_R = self.R.copy()
    #             return
    #         else:
    #             # Opposite direction (180 rotation)
    #             # Find perpendicular vector to use as rotation axis
    #             if abs(c_old[0]) < abs(c_old[1]):
    #                 omega = np.array([0, c_old[2], -c_old[1]])
    #             else:
    #                 omega = np.array([c_old[2], 0, -c_old[0]])
    #             omega = omega / np.linalg.norm(omega)
    #             theta = np.pi # 180 degrees
    #     else:
    #         # Normal case
    #         omega = omega / omega_norm # Normalize rotation axis
    #         # 5. Calculate rotation angle
    #         cos_theta = np.clip(np.dot(c_old, c_new), -1.0, 1.0) # Handle numerical issues
    #         theta = np.arccos(cos_theta)
        
    #     # 6. Skew-symmetric matrix
    #     omega_skew = np.array([
    #         [0, -omega[2], omega[1]],
    #         [omega[2], 0, -omega[0]],
    #         [-omega[1], omega[0], 0]
    #     ])
        
    #     # 7. Rotation matrix using Rodrigues formula
    #     A = np.eye(3) + np.sin(theta) * omega_skew + (1 - np.cos(theta)) * (omega_skew @ omega_skew)
        
    #     # 8 & 9. Update local coordinate system
    #     self.current_R = np.zeros_like(self.R)
    #     self.current_R[0, :] = c_new
    #     self.current_R[1, :] = A @ self.R[1, :]  # Rotate y-axis
    #     self.current_R[2, :] = A @ self.R[2, :]  # Rotate z-axis
        
    #     # 10. Re-orthogonalize 
    #     x_local = self.current_R[0, :]
        
    #     # Adjust y-axis to be perpendicular to x-axis
    #     y_temp = self.current_R[1, :]
    #     y_local = y_temp - np.dot(y_temp, x_local) * x_local
    #     y_local = y_local / np.linalg.norm(y_local)
        
    #     # z-axis completes the orthogonal system
    #     z_local = np.cross(x_local, y_local)
    #     z_local = z_local / np.linalg.norm(z_local)
        
    #     # Updated local coordinate system
    #     self.current_R[0, :] = x_local
    #     self.current_R[1, :] = y_local
    #     self.current_R[2, :] = z_local

    #     self.current_state = {
    #         'coords': [coord.copy() for coord in self.current_coords],
    #         'R': self.current_R.copy(),
    #         'L': self.current_L
    #     }

    # def update_state(self, displacements):
    #     """
    #     Update the element's state based on the provided displacements.
        
    #     Args:
    #         displacements: List of displacements for each node [disp_node1, disp_node2]
    #                     Each node displacement is [dx, dy, dz]
    #     """
    #     # Unpack displacements
    #     disp1, disp2 = displacements
        
    #     # Ensure all displacement components are defined (defaults to 0.0)
    #     disp1 = [disp1[i] if i < len(disp1) and disp1[i] is not None else 0.0 for i in range(3)]
    #     disp2 = [disp2[i] if i < len(disp2) and disp2[i] is not None else 0.0 for i in range(3)]
        
    #     # Store previous state
    #     self.previous_state = self.current_state.copy()
        
    #     # Update current state with new positions
    #     self.current_state['coords'][0] = self.initial_state['coords'][0] + np.array(disp1)
    #     self.current_state['coords'][1] = self.initial_state['coords'][1] + np.array(disp2)
        
    #     # Recalculate element length
    #     delta = self.current_state['coords'][1] - self.current_state['coords'][0]
    #     L = np.linalg.norm(delta)
    #     self.current_state['L'] = L
        
    #     # Update local coordinate system (if length isn't too small)
    #     if L > 1e-10:
    #         # New x-axis (along element)
    #         x_local = delta / L
            
    #         # Get original y-axis from initial state
    #         orig_y = self.initial_state['R'][1]
            
    #         # Project out x component to ensure orthogonality
    #         y_temp = orig_y - np.dot(orig_y, x_local) * x_local
    #         if np.linalg.norm(y_temp) > 1e-10:
    #             # Normalize to get new y-axis
    #             y_local = y_temp / np.linalg.norm(y_temp)
    #         else:
    #             # If y is too close to x, use a fallback direction
    #             fallback = np.array([0, 0, 1]) if abs(x_local[2]) < 0.9 else np.array([1, 0, 0])
    #             y_temp = fallback - np.dot(fallback, x_local) * x_local
    #             y_local = y_temp / np.linalg.norm(y_temp)
            
    #         # New z-axis from cross product
    #         z_local = np.cross(x_local, y_local)
    #         z_local = z_local / np.linalg.norm(z_local)
            
    #         # Create new rotation matrix
    #         R = np.vstack((x_local, y_local, z_local))
    #         self.current_state['R'] = R

    def update_state(self, displacements):
        """
        Update the element's state based on the provided displacements,
        preserving proper rotation behavior.
        
        Args:
            displacements: List of displacements for each node [disp_node1, disp_node2]
                        Each node displacement is [dx, dy, dz]
        """
        # Ensure displacements are valid
        disp1 = np.array([d if d is not None else 0.0 for d in displacements[0][:3]])
        disp2 = np.array([d if d is not None else 0.0 for d in displacements[1][:3]])
        
        # Store previous state
        # self.previous_state = self.current_state.copy()
        
        # Get initial coordinates and rotation matrix
        init_coords = self.initial_state['coords']
        init_R = self.initial_state['R']
        
        # Calculate new coordinates
        new_coords = [
            init_coords[0] + disp1,
            init_coords[1] + disp2
        ]
        
        # Calculate new element vector and length
        delta = new_coords[1] - new_coords[0]
        L = np.linalg.norm(delta)
        
        # Initialize new rotation matrix with initial values
        new_R = init_R.copy()
        
        # Only update orientation if length is not too small
        if L > 1e-10:
            # New x-axis is along the element
            x_new = delta / L
            
            # Get initial x-axis
            x_old = init_R[0]
            
            # Compute rotation axis and angle
            rotation_axis = np.cross(x_old, x_new)
            axis_norm = np.linalg.norm(rotation_axis)
            
            if axis_norm > 1e-10:
                # Normal case - use Rodrigues rotation
                rotation_axis = rotation_axis / axis_norm
                cos_angle = np.clip(np.dot(x_old, x_new), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Create rotation matrix using Rodrigues' formula
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                
                R_matrix = (np.eye(3) + np.sin(angle) * K + 
                            (1 - cos_angle) * np.dot(K, K))
                
                # Apply rotation to all axes
                for i in range(3):
                    new_R[i] = R_matrix @ init_R[i]
                    
            elif np.dot(x_old, x_new) < 0:
                # Special case: 180-degree rotation
                # Find an arbitrary perpendicular vector for rotation axis
                if abs(x_old[0]) < abs(x_old[1]) and abs(x_old[0]) < abs(x_old[2]):
                    perp = np.array([0.0, -x_old[2], x_old[1]])
                else:
                    perp = np.array([-x_old[1], x_old[0], 0.0])
                    
                perp = perp / np.linalg.norm(perp)
                
                # Create rotation matrix for 180 degrees around this axis
                K = np.array([
                    [0, -perp[2], perp[1]],
                    [perp[2], 0, -perp[0]],
                    [-perp[1], perp[0], 0]
                ])
                
                R_matrix = np.eye(3) + 2 * np.dot(K, K)
                
                # Apply rotation to all axes
                for i in range(3):
                    new_R[i] = R_matrix @ init_R[i]
            
            # Ensure the x-axis is exactly along the element
            new_R[0] = x_new
            
            # Re-orthogonalize to ensure a proper rotation matrix
            # y-axis: remove x component and normalize
            new_R[1] = new_R[1] - np.dot(new_R[1], new_R[0]) * new_R[0]
            new_R[1] = new_R[1] / np.linalg.norm(new_R[1])
            
            # z-axis: cross product of x and y
            new_R[2] = np.cross(new_R[0], new_R[1])
            new_R[2] = new_R[2] / np.linalg.norm(new_R[2])
        
        # Update the current state
        self.current_state = {
            'coords': new_coords,
            'R': new_R,
            'L': L
        }

    def reset_state(self):
        """Reset (current and previous state) to initial state."""
        self.current_state = self.initial_state.copy()
        self.previous_state = self.initial_state.copy()
        
        # Also reset any other state variables
        self.current_coords = [node.coords.copy() for node in self.nodes]
        self.current_R = self.R.copy()
        self.current_L = self.L