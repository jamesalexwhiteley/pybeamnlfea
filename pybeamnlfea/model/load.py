import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Load:
    """Base class for loads applied to the structure"""
    def __init__(self, node_id: int):
        self.node_id = node_id

class NodalLoad(Load):
    """Represents a force/moment load applied directly to a node"""
    def __init__(self, node_id: int, force_vector: np.ndarray):
        """
        Initialise a nodal load.    
        
        Args:
            node_id: node tag 
            force_vector: array of 7 values [Fx, Fy, Fz, Mx, My, Mz, Bx] 
        """
        super().__init__(node_id)
        if len(force_vector) != 7:
            raise ValueError("Force vector must have 7 values")
        self.force_vector = np.array(force_vector, dtype=float)
    
    def __repr__(self) -> str:
        """String representation of the load."""
        return f"NodalLoad({self.node_id}, forces={self.force_vector})"