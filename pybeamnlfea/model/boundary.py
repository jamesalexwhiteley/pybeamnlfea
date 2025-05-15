import numpy as np
from typing import List, Dict, Optional, Union, Tuple

# Author: James Whiteley (github.com/jamesalexwhiteley)

class BoundaryCondition:
    """Represents a boundary condition applied to a node, with support for elastic constraints"""
    def __init__(self, node_id: int, dof_constraints: List[bool]):
        """
        Initialise a boundary condition.

        Args:
            node_id: node tag  
            dof_constraints: [ux, uy, uz, θx, θy, θz, φ] where 0 indicates fixed, 1 indicates free

        """
        self.node_id = node_id 
        if len(dof_constraints) != 7:
            raise ValueError("DOF constraints must have 7 values")
        self.dof_constraints = dof_constraints
        
        # Add support for elastic constraints
        # None means not elastically supported, a float value represents the stiffness
        self.elastic_stiffness = [None] * 7
        self.prescribed_displacements = [0.0] * 7
    
    @property
    def constrained_dofs(self) -> List[int]:
        """Returns the global DOF indices that are rigidly constrained (not elastically)"""
        # Only include DOFs that are constrained (dof_constraint=False) and not elastic
        return [i for i, free in enumerate(self.dof_constraints) if not free and self.elastic_stiffness[i] is None]
    
    def is_dof_constrained(self, i):
        """Returns True if index is rigidly constrained (not free and not elastic)"""
        if i >= len(self.dof_constraints):
            return False
        return i in self.constrained_dofs
    
    def add_elastic_support(self, dof_index: int, stiffness: float, prescribed_displacement: float = 0.0):
        """
        Add an elastic support with finite stiffness for a specific DOF.
        
        Args:
            dof_index: index of the DOF (0=ux, 1=uy, 2=uz, 3=θx, 4=θy, 5=θz, 6=φ)
            stiffness: stiffness value of the elastic support
            prescribed_displacement: optional prescribed displacement/rotation
        """
        if dof_index >= len(self.dof_constraints):
            raise ValueError(f"DOF index {dof_index} is out of range (0-6)")
        
        # Set as elastically supported
        self.dof_constraints[dof_index] = True  # Mark as "free" in original constraint system
        self.elastic_stiffness[dof_index] = stiffness
        self.prescribed_displacements[dof_index] = prescribed_displacement
    
    def has_finite_stiffness(self, dof_index: int) -> bool:
        """Check if a DOF has an elastic support with finite stiffness."""
        if dof_index >= len(self.elastic_stiffness):
            return False
        return self.elastic_stiffness[dof_index] is not None
    
    def get_support_stiffness(self, dof_index: int) -> float:
        """Get the stiffness value for an elastic support."""
        if dof_index >= len(self.elastic_stiffness) or self.elastic_stiffness[dof_index] is None:
            return 0.0
        return self.elastic_stiffness[dof_index]
    
    def has_prescribed_displacement(self, dof_index: int) -> bool:
        """Check if a DOF has a prescribed displacement/rotation."""
        if dof_index >= len(self.prescribed_displacements) or not self.has_finite_stiffness(dof_index):
            return False
        return abs(self.prescribed_displacements[dof_index]) > 1e-10  # Small threshold to account for floating-point
    
    def get_prescribed_displacement(self, dof_index: int) -> float:
        """Get the prescribed displacement/rotation for a DOF."""
        if dof_index >= len(self.prescribed_displacements) or not self.has_finite_stiffness(dof_index):
            return 0.0
        return self.prescribed_displacements[dof_index]
    
    def __repr__(self) -> str:
        """String representation of boundary condition."""
        elastic_info = []
        for i, stiffness in enumerate(self.elastic_stiffness):
            if stiffness is not None:
                elastic_info.append(f"DOF{i}(k={stiffness}, u={self.prescribed_displacements[i]})")
        
        elastic_str = ", elastic: [" + ", ".join(elastic_info) + "]" if elastic_info else ""
        return f"BC({self.node_id}, dofs={self.dof_constraints}{elastic_str})"