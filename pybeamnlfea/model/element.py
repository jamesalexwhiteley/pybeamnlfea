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
        # # Compute local x axis along member direction
        # a = self.nodes[1].coords - self.nodes[0].coords
        # a = a / np.linalg.norm(a)
        
        # # Global z axis
        # global_z = np.array([0.0, 0.0, 1.0])
        
        # if np.abs(np.dot(a, global_z)) > 0.95:
        #     # If nearly parallel to global z, use global y instead
        #     ref_vector = np.array([0.0, 1.0, 0.0])
        # else:
        #     ref_vector = global_z
        
        # # Compute local z axis perpendicular to local x
        # c = np.cross(a, ref_vector) 
        # c = c / np.linalg.norm(c)
        
        # # Compute local y axis to complete right-handed system
        # b = np.cross(c, a) 
        # b = b / np.linalg.norm(b)

        # self.R = np.vstack((a, b, c))



        a = self.nodes[1].coords - self.nodes[0].coords
        x_local = a / np.linalg.norm(a)

        # x1, y1, z1 = self.nodes[n1]
        # x2, y2, z2 = self.nodes[n2]
        # dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)

        # L = np.sqrt(dx*dx + dy*dy + dz*dz)
        # if L < 1e-14:
        #     return np.eye(3)

        # # local x-axis
        # x_local = np.array([dx, dy, dz]) / L

        # "reference up" vector
        ref_up = np.array([0, 0, 1], dtype=float)

        # try cross(x_local, ref_up)
        y_l_temp = np.cross(x_local, ref_up)
        nrm = np.linalg.norm(y_l_temp)
        if nrm < 1e-12:
            # fallback
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

        R = np.vstack([x_local, y_local, z_local])
        self.R = R

    def compute_local_stiffness_matrix(self):
        """Compute local stiffness matrix."""

        # elastic stiffness components 
        A = self.section.A
        Iy = self.section.Iy
        Iz = self.section.Iz
        J = self.section.J
        Iw = self.section.Iw
        
        node0, node1 = self.nodes[0], self.nodes[1]
        E = self.material.E
        G = self.material.G
        L = node0.distance_to(node1)
        self.length = L

        # geoemtric stiffness components 
        P0=0
        My0=0
        Mz0=0
        B0_bar=0

        W_bar=0
        y0=0
        z0=0
        beta_y=0
        beta_z=0
        r=0
        
        self.k = thin_wall_stiffness_matrix(E, G, A, Iy, Iz, Iw, J, L, 
                                            P0, My0, Mz0, B0_bar, 
                                            W_bar, y0, z0, beta_y, beta_z, r)

        return self.k
    
    def get_local_to_global_transformation_matrix(self):
        """Compute the transformation matrix for this element."""
        
        # Rotation matrix  
        core_block = np.zeros((7, 7))
        core_block[:3, :3] = self.R
        core_block[3:6, 3:6] = self.R
        core_block[6, 6] = 1
        
        I2 = np.eye(2)
        Q = np.kron(I2, core_block)
        
        return Q
    
    def get_controid_transformation_matrix(self): 
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
        C = self.get_controid_transformation_matrix()
        Q = self.get_local_to_global_transformation_matrix()
        self.T = np.linalg.inv(C @ Q @ np.linalg.inv(C))                                  
 
        return self.T