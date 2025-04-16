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
        self.nodes = nodes
        self.material = material
        self.section = section
        self.local_axes = self._initialise_local_axes()

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
        """
        Compute the transformation matrix T = (CQC^-1)^-1 
        """
        # C = self.compute_controid_transformation_matrix()
        # Q = self.compute_local_to_global_transformation_matrix()
        # self.T = np.linalg.inv(C @ Q @ np.linalg.inv(C))   
        self.T = self.compute_local_to_global_transformation_matrix()                       # NOTE                  
 
        return self.T