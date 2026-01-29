import numpy as np 
from pybeamnlfea.model.node import Node 
from pybeamnlfea.model.material import Material 
from pybeamnlfea.model.section import Section 
from pybeamnlfea.utils.stiffness_matrix import thin_wall_stiffness_matrix_bazant, thin_wall_stiffness_matrix_chan, thin_wall_stiffness_matrix_derived

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
        
        # Need to track deformation as analysis progresses 
        self.current_state = self.initial_state.copy()

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

    def _get_section_geometry(self):
        """
        Extract section geometry parameters with defaults.
        
        Returns dict with y0, z0, beta_y, beta_z, beta_w, r1
        """
        # Shear center coordinates (default to centroid)
        y0 = getattr(self.section, 'y0', 0.0)
        z0 = getattr(self.section, 'z0', 0.0)
        
        # Wagner coefficients / asymmetry parameters
        beta_y = getattr(self.section, 'beta_y', 0.0)
        beta_z = getattr(self.section, 'beta_z', 0.0)
        beta_w = getattr(self.section, 'beta_w', 0.0)
        
        # Polar radius of gyration squared
        # r1^2 = (Iy + Iz)/A + y0^2 + z0^2
        if hasattr(self.section, 'r1'):
            r1 = self.section.r1
        else:
            r1 = (self.section.Iy + self.section.Iz) / self.section.A + y0**2 + z0**2
            
        return {
            'y0': y0,
            'z0': z0,
            'beta_y': beta_y,
            'beta_z': beta_z,
            'beta_w': beta_w,
            'r1': r1
        }

    def compute_elastic_stiffness_matrix(self): 
        """
        Compute local elastic stiffness matrix (without geometric effects).
        
        Returns:
            scipy.sparse.csr_matrix: 14x14 elastic stiffness matrix
        """
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
        k_elastic = thin_wall_stiffness_matrix_bazant(
            E, G, A, Iy, Iz, Iw, J, L,
        )
        
        return k_elastic

    def compute_local_stiffness_matrix(self, include_geometric=False, internal_forces=None):
        """
        Compute local stiffness matrix with optional geometric effects.

        """
        # Section and material properties
        A = self.section.A
        Iy = self.section.Iy
        Iz = self.section.Iz
        J = self.section.J
        Iw = self.section.Iw
        
        E = self.material.E
        G = self.material.G
        L = self.L
        
        # Get section geometry
        geom = self._get_section_geometry()
        
        # Force parameters
        P = 0
        My1 = My2 = 0
        Mz1 = Mz2 = 0
        Mw = 0
        load_height = q = 0
        
        if include_geometric and internal_forces is not None:
            # Extract forces
            P = internal_forces.get('axial', internal_forces.get('P', 0))
            
            My1 = internal_forces.get('moment_y1', internal_forces.get('My1', 0))
            My2 = internal_forces.get('moment_y2', internal_forces.get('My2', 0))
            Mz1 = internal_forces.get('moment_z1', internal_forces.get('Mz1', 0))
            Mz2 = internal_forces.get('moment_z2', internal_forces.get('Mz2', 0))
            
            Mw = internal_forces.get('bimoment', internal_forces.get('Mw', 0))

            # Get UDL load height effect if present
            load_height = getattr(self, 'load_height', 0)
            q = getattr(self, 'q_transverse', 0)
        
        # # Combined stiffness matrix
        # k = thin_wall_stiffness_matrix_chan(
        #     E, G, A, Iy, Iz, Iw, J, L,
        #     P=P, My1=My1, My2=My2, Mz1=Mz1, Mz2=-Mz2, Mw=Mw,
        #     y0=geom['y0'], z0=geom['z0'],
        #     beta_y=geom['beta_y'], beta_z=geom['beta_z'], beta_w=geom['beta_w'],
        #     r1=geom['r1'],
        #     include_geometric=include_geometric
        # )

        # k = thin_wall_stiffness_matrix_bazant(
        #         E, G, A, Iy, Iz, Iw, J, L, 
        #         # P0=P, My0=(My1+My2)/2, Mz0=(Mz1+Mz2)/2, B0_bar=Mw,
        #         P0=P, Mz0=(My1+My2), My0=(Mz1+Mz2), B0_bar=Mw,
        #         W_bar=0, y0=geom['y0'], z0=geom['z0'], beta_y=geom['beta_y'], beta_z=geom['beta_z'], r=geom['r1']
        # )  

        k = thin_wall_stiffness_matrix_derived(
            E, G, A, Iy, Iz, Iw, J, L,
            P=P, My1=My1, My2=My2, Mz1=Mz1, Mz2=-Mz2, Mw=Mw,
            y0=geom['y0'], z0=geom['z0'],
            beta_y=geom['beta_y'], beta_z=geom['beta_z'], beta_w=geom['beta_w'],
            r1=geom['r1'],
            load_height=load_height, q=q,
            include_geometric=include_geometric
        )
        
        return k

    def compute_internal_forces(self, local_displacements):
        """
        Compute internal forces from element local displacements.
        
        Args:
            local_displacements : array-like
                14-element array of local nodal displacements
                [u1, v1, w1, θx1, θy1, θz1, φ1, u2, v2, w2, θx2, θy2, θz2, φ2]
            
        """
        import numpy as np
        local_displacements = np.asarray(local_displacements).flatten()
        
        # Elastic stiffness matrix
        k_elastic = self.compute_elastic_stiffness_matrix()
        
        # Internal forces vector: F = K * u
        internal_forces_vector = k_elastic @ local_displacements
        
        # DOF ordering: [u, v, w, θx, θy, θz, φ] per node
        # Indices:       0  1  2   3   4   5  6  (node 1)
        #                7  8  9  10  11  12 13  (node 2)
        
        # Extract forces at each node
        # Axial forces
        P1 = internal_forces_vector[0]    # Axial force at node 1
        P2 = -internal_forces_vector[7]   # Axial force at node 2 
        
        # Moments about y-axis (bending in x-z plane)
        My1 = internal_forces_vector[4]  # Moment about y at node 1
        My2 = -internal_forces_vector[11] # Moment about y at node 2
        
        # Moments about z-axis (bending in x-y plane)
        Mz1 = internal_forces_vector[5]   # Moment about z at node 1
        Mz2 = -internal_forces_vector[12]  # Moment about z at node 2 
        
        # Bimoments (warping)
        B1 = internal_forces_vector[6]    # Bimoment at node 1
        B2 = internal_forces_vector[13]   # Bimoment at node 2
        
        # Average values 
        P = (P1 + P2) / 2
        Mw = (B1 + B2) / 2  
        
        return {
            'axial': P,
            'P': P,
            'moment_y1': My1,
            'moment_y2': My2,
            'My1': My1,
            'My2': My2,
            'moment_z1': Mz1,
            'moment_z2': Mz2,
            'Mz1': Mz1,
            'Mz2': Mz2,
            'bimoment': Mw,
            'Mw': Mw,
            'P1': P1, 'P2': P2,
            'B1': B1, 'B2': B2
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

    def compute_centroidal_transformation_matrix(self): 
        """
        Compute the shear center -> centroid transformation matrix t
        Based on Bazant 1973, Eq. 25-28
    
        """
        t = np.eye(14)  
        
        y0 = getattr(self.section, 'y0', 0.0)
        z0 = getattr(self.section, 'z0', 0.0)
        
        # Node 1 (DOFs 0-6)
        t[1, 3] = -z0   # v_centroid = v_shear - z0 * θx
        t[2, 3] = y0    # w_centroid = w_shear + y0 * θx
        
        # Node 2 (DOFs 7-13)
        t[8, 10] = -z0
        t[9, 10] = y0
        
        return t

    def compute_transformation_matrix(self):
        """
        Compute the full transformation matrix T = t @ Q @ t^(-1)
        Based on Bazant 1973, Eq. 27
        """
        t = self.compute_centroidal_transformation_matrix()
        Q = self.compute_local_to_global_transformation_matrix()
        t_inv = np.linalg.inv(t)
        
        self.T = t @ Q @ t_inv
        # self.T = np.eye((14))

        return self.T

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
        self.current_coords = [node.coords.copy() for node in self.nodes]
        self.current_R = self.R.copy()
        self.current_L = self.L