import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy.sparse.linalg import spsolve

from node import Node 
from element import ThinWalledBeamElement
from material import Material
from section import Section
from assembly import Assembler
from boundary import BoundaryCondition

# Author: James Whiteley (github.com/jamesalexwhiteley)

class ThinWalledFrame:
    """
    Main class for structural FEA model that coordinates nodes, elements, 
    materials, sections, boundary conditions, and loads.
    """
    def __init__(self):
        self.nodes: List[Node] = []
        self.elements: List[ThinWalledBeamElement] = []
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.boundary_conditions: List[BoundaryCondition] = []
        self.loads: Dict[int, np.ndarray] = {}  # node_id -> force/moment vector
        self.assembler: Optional[Assembler] = None
        self.results: Dict = {}
        
    def add_node(self, node_id: int, x: float, y: float, z: float) -> Node:
        """Add a node to the frame."""
        node = Node(node_id, np.array([x, y, z]))
        self.nodes.append(node)
        return node
        
    def add_material(self, name: str, young_modulus: float, poisson_ratio: float, **kwargs) -> Material:
        """Add a material to the frame."""
        material = Material(young_modulus, poisson_ratio, **kwargs)
        self.materials[name] = material
        return material
        
    def add_section(self, name: str, section_properties: Dict) -> Section:
        """Add a cross-section to the frame."""
        section = Section(**section_properties)
        self.sections[name] = section
        return section
        
    def add_element(self, node_ids: List[int], material_name: str, section_name: str) -> ThinWalledBeamElement:
        """Add an element to the frame, connecting specified nodes with given properties."""
        # Find the nodes by ID
        element_nodes = [next(node for node in self.nodes if node.id == node_id) for node_id in node_ids]
        material = self.materials[material_name]
        section = self.sections[section_name]
        
        element = ThinWalledBeamElement(element_nodes, material, section)
        self.elements.append(element)
        return element
        
    def add_boundary_condition(self, node_id: int, dof_restraints: Dict[int, float]) -> None:
        """
        Add boundary condition at a node.
        dof_restraints: Dictionary mapping DOF indices (0-5) to prescribed values (usually 0 for fixed)
        """
        node = next(node for node in self.nodes if node.id == node_id)
        bc = BoundaryCondition(node, dof_restraints)
        self.boundary_conditions.append(bc)
        
    def add_load(self, node_id: int, force_vector: List[float]) -> None:
        """Add a load (force/moment) at a node."""
        if len(force_vector) != 6:
            raise ValueError("Force vector must have 6 components [Fx, Fy, Fz, Mx, My, Mz]")
        self.loads[node_id] = np.array(force_vector)
        
    def initialize_assembly(self) -> None:
        """Initialize the assembler with current elements and nodes."""
        self.assembler = Assembler(self.elements, self.nodes)
        
    # def solve(self, solver_type: str = "linear_static") -> Dict:
    #     """Solve the FEA model and store results."""
    #     if not self.assembler:
    #         self.initialize_assembly()
            
    #     num_dofs = len(self.nodes) * 6  # 6 DOFs per node
        
    #     # Assemble global stiffness matrix
    #     K_global = self.assembler.assemble_global_stiffness_matrix(self.elements, num_dofs)
        
    #     # Create global force vector
    #     F_global = np.zeros(num_dofs)
    #     for node_id, force in self.loads.items():
    #         dof_indices = [node_id * 6 + i for i in range(6)]
    #         F_global[dof_indices] += force
            
    #     # Apply boundary conditions
    #     # This is a simplified approach - you might want to implement a more sophisticated method
    #     free_dofs = list(range(num_dofs))
    #     fixed_values = np.zeros(num_dofs)
        
    #     for bc in self.boundary_conditions:
    #         node_id = bc.node.id
    #         for dof, value in bc.dof_restraints.items():
    #             global_dof = node_id * 6 + dof
    #             if global_dof in free_dofs:
    #                 free_dofs.remove(global_dof)
    #             fixed_values[global_dof] = value
                
    #     # Solve the system (for free DOFs only)
    #     K_free = K_global[np.ix_(free_dofs, free_dofs)]
    #     F_free = F_global[free_dofs]
        
    #     # Adjust for fixed DOFs' contribution
    #     for bc in self.boundary_conditions:
    #         node_id = bc.node.id
    #         for dof, value in bc.dof_restraints.items():
    #             global_dof = node_id * 6 + dof
    #             if value != 0:  # Non-zero prescribed displacement
    #                 F_free -= K_global[np.ix_(free_dofs, [global_dof])] * value
        
    #     # Solve for displacements
    #     displacements_free = spsolve(K_free, F_free)
        
    #     # Reconstruct full displacement vector
    #     displacements = fixed_values.copy()
    #     for i, dof in enumerate(free_dofs):
    #         displacements[dof] = displacements_free[i]
            
    #     # Store results
    #     self.results = {
    #         "displacements": displacements,
    #         "reaction_forces": self._calculate_reaction_forces(K_global, displacements)
    #     }
        
    #     # Calculate element internal forces
    #     self._calculate_internal_forces(displacements)
        
    #     return self.results
        
    # def _calculate_reaction_forces(self, K_global, displacements):
    #     """Calculate reaction forces at supports."""
    #     reaction_forces = K_global.dot(displacements)
        
    #     # Only return reactions at constrained DOFs
    #     reactions = {}
    #     for bc in self.boundary_conditions:
    #         node_id = bc.node.id
    #         node_reactions = []
    #         for dof in range(6):
    #             global_dof = node_id * 6 + dof
    #             if dof in bc.dof_restraints:  # If this DOF is constrained
    #                 node_reactions.append(reaction_forces[global_dof])
    #             else:
    #                 node_reactions.append(0.0)
    #         reactions[node_id] = np.array(node_reactions)
            
    #     return reactions
        
    # def _calculate_internal_forces(self, global_displacements):
    #     """Calculate internal forces for all elements."""
    #     element_forces = {}
        
    #     for i, elem in enumerate(self.elements):
    #         # Get global DOFs for this element
    #         global_dofs = []
    #         for node in elem.nodes:
    #             global_dofs.extend([node.id * 6 + j for j in range(6)])
                
    #         # Extract element displacements from global vector
    #         elem_displacements = global_displacements[global_dofs]
            
    #         # Calculate internal forces
    #         internal_forces = elem.internal_forces(elem_displacements)
    #         element_forces[i] = internal_forces
            
    #     self.results["element_forces"] = element_forces
        
    # def get_displacement_at_node(self, node_id: int) -> np.ndarray:
    #     """Get displacement vector at a specific node."""
    #     if not self.results or "displacements" not in self.results:
    #         raise RuntimeError("Solve the model first before accessing results")
            
    #     start_idx = node_id * 6
    #     return self.results["displacements"][start_idx:start_idx + 6]
        
    # def get_reaction_at_node(self, node_id: int) -> np.ndarray:
    #     """Get reaction force/moment vector at a specific node."""
    #     if not self.results or "reaction_forces" not in self.results:
    #         raise RuntimeError("Solve the model first before accessing results")
            
    #     if node_id not in self.results["reaction_forces"]:
    #         return np.zeros(6)  # No reaction at unconstrained nodes
            
    #     return self.results["reaction_forces"][node_id]
        
    # def get_element_forces(self, element_idx: int) -> Dict:
    #     """Get internal forces for a specific element."""
    #     if not self.results or "element_forces" not in self.results:
    #         raise RuntimeError("Solve the model first before accessing results")
            
    #     return self.results["element_forces"].get(element_idx, None)