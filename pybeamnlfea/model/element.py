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

    # def _initialise_local_axes(self):
    #     """Compute local x, y, z axes based on element geometry."""

    #     # Compute local z axis 
    #     c1 = self.nodes[1].coords - self.nodes[0].coords
    #     c = c1 / np.linalg.norm(c1)  
        
    #     # Find a vector not parallel to c to use for cross product
    #     if abs(c[0]) < abs(c[1]) and abs(c[0]) < abs(c[2]):
    #         temp = np.array([1, 0, 0])
    #     elif abs(c[1]) < abs(c[2]):
    #         temp = np.array([0, 1, 0])
    #     else:
    #         temp = np.array([0, 0, 1])
        
    #     # Compute local x axis 
    #     a1 = np.cross(temp, c)
    #     a = a1 / np.linalg.norm(a1)
        
    #     # Compute local x axis 
    #     b1 = np.cross(c, a)
    #     b = b1 / np.linalg.norm(b1)
        
    #     # Return the three unit vectors as a rotation matrix from local to globals
    #     self.R = np.vstack((a, b, c))

    def _initialise_local_axes(self):
        """
        Compute local coordinate system
            - local z axis is along the member direction
            - local x and y axes are derived from projections of global x and y
        """
        # Compute local z axis along member direction
        c = self.nodes[1].coords - self.nodes[0].coords
        c = c / np.linalg.norm(c)
        
        # Global x axis
        global_x = np.array([1.0, 0.0, 0.0])
        
        if np.abs(np.dot(c, global_x)) > 0.95:
            # If nearly parallel to global x, use global y instead
            ref_vector = np.array([0.0, 1.0, 0.0])
        else:
            ref_vector = global_x
        
        # Compute local y axis perpendicular to local z
        b = np.cross(ref_vector, c)
        b = b / np.linalg.norm(b)
        
        # Compute local x axis to complete right-handed system
        a = np.cross(c, b)
        a = a / np.linalg.norm(a)
        
        # Create rotation matrix from local to global
        self.R = np.vstack((a, b, c))

    def compute_local_stiffness_matrix(self):
        """Compute local stiffness matrix."""

        # elastic stiffness components 
        A = self.section.A
        Ix = self.section.Ix
        Iy = self.section.Iy
        J = self.section.J
        Iw = self.section.Iw
        
        node0, node1 = self.nodes[0], self.nodes[1]
        E = self.material.E
        G = self.material.G
        L = node0.distance_to(node1)
        self.length = L

        # geoemtric stiffness components 
        P0=0
        Mx0=0
        My0=0
        B0_bar=0

        W_bar=0
        x0=0
        y0=0
        beta_x=0
        beta_y=0
        r=0
        
        self.k = thin_wall_stiffness_matrix(E, G, A, Ix, Iy, Iw, J, L, 
                                            P0, Mx0, My0, B0_bar, 
                                            W_bar, x0, y0, beta_x, beta_y, r)
        
        return self.k
    
    def get_local_to_global_transformation_matrix(self):
        """Compute the transformation matrix for this element."""
        
        # Create the core transformation block 
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
        
        # Get the offset between shear center and centroid
        x0 = self.section.x0  
        y0 = self.section.y0  
        
        # Set the specific non-zero off-diagonal terms 
        C[2, 3] = x0 
        C[4, 6] = x0 
        C[9, 10] = x0 
        C[11, 13] = x0  
        
        C[1, 3] = y0 
        C[5, 6] = y0  
        C[8, 10] = y0  
        C[12, 13] = y0  
        
        return C

    def compute_transformation_matrix(self):
        """
        Compute the transformation matrix T = (CQC^-1)^-1 
        """
        C = self.get_controid_transformation_matrix()
        Q = self.get_local_to_global_transformation_matrix()
        self.T = np.linalg.inv(C @ Q @ np.linalg.inv(C))                                     
 
        return self.T
        
    # def internal_forces(self, displacements):
    #     # Calculate internal forces given displacements
    #     pass

    # def update_local_axes(self, displacements):
    #     # Update local coordinate system based on current deformation
    #     pass