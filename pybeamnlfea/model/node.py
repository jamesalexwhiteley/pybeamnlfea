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
        
        # # Initial displacements/rotations/warping are zero
        # # [w, ū, v̄, θz, -θ̄x, θ̄y, φ]
        # self.displacements = np.zeros(7)
        
        # # For storing applied loads at this node (corresponding to each DOF)
        # self.applied_loads = np.zeros(7)
        
        # # For storing reactions at this node (if it's a support)
        # self.reactions = np.zeros(7)
        
        # # For tracking boundary conditions applied to this node
        # self.constrained_dofs: Dict[int, float] = {}  # dof_index -> prescribed value
        
        # Dictionary to map DOF indices to their physical meaning
        self.dof_names = {
            0: "w",       # translation along local z
            1: "ū",       # translation along local x
            2: "v̄",       # translation along local y
            3: "θz",      # rotation about local z
            4: "-θ̄x",     # negative rotation about local x
            5: "θ̄y",      # rotation about local y
            6: "φ"        # warping (dθz/dz)
        }

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
        
        Args:
            other_node: Another Node object
            
        Returns:
            Float: Euclidean distance between the two nodes
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