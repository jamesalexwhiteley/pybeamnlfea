import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# Author: James Whiteley (github.com/jamesalexwhiteley)

class BoundaryCondition:
    """Represents a boundary condition applied to a node"""
    def __init__(self, node_id: int, dof_constraints: List[bool]):
        """
        Initialise a boundary condition.

        Args:
            node_id: node tag  
            dof_constraints: [ux, uy, uz, θx, θy, θz, φ] where 0 indicates fixed

        """
        self.node_id = node_id 
        if len(dof_constraints) != 7:
            raise ValueError("DOF constraints must have 7 values")
        self.dof_constraints = dof_constraints
    
    @property
    def constrained_dofs(self) -> List[int]:
        """Returns the global DOF indices that are constrained"""
        return [i for i, free in enumerate(self.dof_constraints) if not free]
    
    def is_dof_constrained(self, i):
        """Returns True if indice is constrained"""
        return i in self.constrained_dofs
    
    def __repr__(self) -> str:
        """String representation of boundary condition."""
        return f"BC({self.node_id}, dofs={self.dof_constraints})"