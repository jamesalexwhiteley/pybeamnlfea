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
    def __init__(self, id: int, coordinates: np.ndarray):
        """
        Initialize a node with an ID and coordinates.
        
        Args:
            id: Unique identifier for the node
            coordinates: NumPy array [x, y, z] defining node position
        """
        self.id = id
        self.coordinates = np.array(coordinates, dtype=float)
        if len(self.coordinates) != 3:
            raise ValueError("Node coordinates must be vector [x, y, z]")
        
        # Initial displacements/rotations/warping are zero
        # [w, ū, v̄, θz, -θ̄x, θ̄y, φ]
        self.displacements = np.zeros(7)
        
        # For storing applied loads at this node (corresponding to each DOF)
        self.applied_loads = np.zeros(7)
        
        # For storing reactions at this node (if it's a support)
        self.reactions = np.zeros(7)
        
        # For tracking boundary conditions applied to this node
        self.constrained_dofs: Dict[int, float] = {}  # dof_index -> prescribed value
        
        # Dictionary to map DOF indices to their physical meaning
        self.dof_map = {
            0: "w",       # translation along local z
            1: "ū",       # translation along local x
            2: "v̄",       # translation along local y
            3: "θz",      # rotation about local z
            4: "-θ̄x",     # negative rotation about local x
            5: "θ̄y",      # rotation about local y
            6: "φ"        # warping (dθz/dz)
        }
    
    @property
    def x(self) -> float:
        """X-coordinate of the node."""
        return self.coordinates[0]
    
    @property
    def y(self) -> float:
        """Y-coordinate of the node."""
        return self.coordinates[1]
    
    @property
    def z(self) -> float:
        """Z-coordinate of the node."""
        return self.coordinates[2]
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node({self.id}, coords={self.coordinates})"
    
    def set_coordinates(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> None:
        """
        Update node coordinates.
        
        Args:
            x: New x-coordinate (if None, keeps existing value)
            y: New y-coordinate (if None, keeps existing value)
            z: New z-coordinate (if None, keeps existing value)
        """
        if x is not None:
            self.coordinates[0] = x
        if y is not None:
            self.coordinates[1] = y
        if z is not None:
            self.coordinates[2] = z
    
    def distance_to(self, other_node: 'Node') -> float:
        """
        Calculate Euclidean distance to another node.
        
        Args:
            other_node: Another Node object
            
        Returns:
            Float: Euclidean distance between the two nodes
        """
        return np.linalg.norm(self.coordinates - other_node.coordinates)
    
    # def apply_load(self, load_vector: List[float]) -> None:
    #     """
    #     Apply a load to the node.
        
    #     Args:
    #         load_vector: List of 6 components [Fx, Fy, Fz, Mx, My, Mz]
    #     """
    #     if len(load_vector) != 6:
    #         raise ValueError("Load vector must have 6 components [Fx, Fy, Fz, Mx, My, Mz]")
    #     self.applied_loads = np.array(load_vector)
    
    # def add_load(self, load_vector: List[float]) -> None:
    #     """
    #     Add a load to any existing loads on the node.
        
    #     Args:
    #         load_vector: List of 6 components [Fx, Fy, Fz, Mx, My, Mz]
    #     """
    #     if len(load_vector) != 6:
    #         raise ValueError("Load vector must have 6 components [Fx, Fy, Fz, Mx, My, Mz]")
    #     self.applied_loads += np.array(load_vector)
    
    # def constrain_dof(self, dof_index: int, value: float = 0.0) -> None:
    #     """
    #     Constrain a degree of freedom to a specified value (default 0).
        
    #     Args:
    #         dof_index: Index of DOF to constrain (0-5)
    #         value: Prescribed value (default 0.0 for fixed constraint)
    #     """
    #     if dof_index not in range(6):
    #         raise ValueError("DOF index must be between 0 and 5")
    #     self.constrained_dofs[dof_index] = value
    
    # def constrain_translation(self, direction: str, value: float = 0.0) -> None:
    #     """
    #     Constrain translational DOF using x/y/z direction.
        
    #     Args:
    #         direction: 'x', 'y', or 'z'
    #         value: Prescribed displacement value
    #     """
    #     dof_map = {'x': 0, 'y': 1, 'z': 2}
    #     if direction.lower() not in dof_map:
    #         raise ValueError("Direction must be 'x', 'y', or 'z'")
    #     self.constrain_dof(dof_map[direction.lower()], value)
    
    # def constrain_rotation(self, axis: str, value: float = 0.0) -> None:
    #     """
    #     Constrain rotational DOF using x/y/z axis.
        
    #     Args:
    #         axis: 'x', 'y', or 'z'
    #         value: Prescribed rotation value
    #     """
    #     dof_map = {'x': 3, 'y': 4, 'z': 5}
    #     if axis.lower() not in dof_map:
    #         raise ValueError("Axis must be 'x', 'y', or 'z'")
    #     self.constrain_dof(dof_map[axis.lower()], value)
    
    # def fix_all_dofs(self) -> None:
    #     """Fix all degrees of freedom (create a fully fixed support)."""
    #     for i in range(6):
    #         self.constrain_dof(i, 0.0)
    
    # def is_constrained(self) -> bool:
    #     """Check if the node has any constrained DOFs."""
    #     return len(self.constrained_dofs) > 0
    
    # def get_global_dof_indices(self) -> List[int]:
    #     """
    #     Get global DOF indices for this node.
        
    #     Returns:
    #         List of 6 indices corresponding to the node's DOFs in the global system
    #     """
    #     return [self.id * 6 + i for i in range(6)]
    
    # def update_displacements(self, displacement_vector: np.ndarray) -> None:
    #     """
    #     Update node displacements from solution vector.
        
    #     Args:
    #         displacement_vector: Array with 6 components [dx, dy, dz, rx, ry, rz]
    #     """
    #     if len(displacement_vector) != 6:
    #         raise ValueError("Displacement vector must have 6 components")
    #     self.displacements = np.array(displacement_vector)
    
    # def update_reactions(self, reaction_vector: np.ndarray) -> None:
    #     """
    #     Update node reactions from solution.
        
    #     Args:
    #         reaction_vector: Array with 6 components [Rx, Ry, Rz, Mx, My, Mz]
    #     """
    #     if len(reaction_vector) != 6:
    #         raise ValueError("Reaction vector must have 6 components")
    #     self.reactions = np.array(reaction_vector)
    
    # def get_deformed_coordinates(self) -> np.ndarray:
    #     """
    #     Get coordinates after applying displacements.
        
    #     Returns:
    #         NumPy array with deformed [x, y, z] coordinates
    #     """
    #     return self.coordinates + self.displacements[:3]
