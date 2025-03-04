import numpy as np 
from pybeamnlfea.model.node import Node 
from pybeamnlfea.model.material import Material 
from pybeamnlfea.model.section import Section 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Element:
    """
    Base element class. 
    """
    def __init__(self, **kwargs):
        """
        Initialize a generic element.
        
        Parameters
        ----------
        **kwargs : dict
            Material properties
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
        # self.local_axes = self._initialize_local_axes()

    def __repr__(self) -> str:
        """String representation of the element."""
        return f"Element({self.id}, nodes={self.nodes})"
        
    # def local_stiffness_matrix(self):
    #     # Compute local stiffness matrix 
    #     pass
        
    # def update_local_axes(self, displacements):
    #     # Update local coordinate system based on current deformation
    #     pass
        
    # def internal_forces(self, displacements):
    #     # Calculate internal forces given displacements
    #     pass

    # def compute_local_axes(node_coords):
    #     # Compute local x, y, z axes based on element geometry
    #     v1 = node_coords[1] - node_coords[0]  # Element axis
    #     x_local = v1 / np.linalg.norm(v1)
        
    #     # Find perpendicular vectors for y and z
    #     # ...