import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Node:
    """
    Node class.
    - 3 translational DOFs (w, ū, v̄) displacements in local system
    - 3 rotational DOFs (θz, -θ̄x, θ̄y) rotations about local system
    - 1 warping DOF (φ = dθz/dz) cross section warping
    """
    def __init__(self, id: int, coords: np.ndarray):
        """
        Initialise a node with an ID and coordinates.
        
        Args:
            id: identifier for the node
            coordinates: NumPy array [x, y, z] defining node position
        """
        self.id = id
        self.coords = np.array(coords, dtype=float)
        self.dof_count = 7
        if len(self.coords) != 3:
            raise ValueError("Node coordinates must be vector [x, y, z]")

    @property
    def ndof(self) -> float:
        """Get number of DOFs."""
        return self.dof_count 
    
    @property
    def x(self) -> float:
        """X-coordinate of the node."""
        return self.coords[0]
    
    @property
    def y(self) -> float:
        """Y-coordinate of the node."""
        return self.coords[1]
    
    @property
    def z(self) -> float:
        """Z-coordinate of the node."""
        return self.coords[2]
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node({self.id}, coords={self.coords})"
    
    def distance_to(self, other_node: 'Node') -> float:
        """
        Calculate Euclidean distance to another node.
        """
        return np.linalg.norm(self.coords - other_node.coords)
    
    def set_coords(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> None:
        """
        Update node coordinates.
        
        Args:
            x: New x-coordinate (if None, keep existing value)
            y: New y-coordinate (if None, keep existing value)
            z: New z-coordinate (if None, keep existing value)
        """
        if x is not None:
            self.coords[0] = x
        if y is not None:
            self.coords[1] = y
        if z is not None:
            self.coords[2] = z