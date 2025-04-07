import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pybeamnlfea.model.node import Node 
from pybeamnlfea.model.material import Material 
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import Element 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import Load 

from pybeamnlfea.solver.assembly import Assembler 
from pybeamnlfea.solver.linear import LinearSolver 
from pybeamnlfea.postprocess.results import Results

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Frame:
    """
    Structural frame FEA model. 
    """
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}
        self.boundary_conditions: Dict[int, BoundaryCondition] = {}
        self.loads: Dict[int, Load] = {}  
        
    def add_node(self, x: float, y: float, z: float, node_id: int = None) -> Node:
        """Add a node to the frame."""
        if node_id is None:
            # Find the next available ID
            node_id = 0 if not self.nodes else max(self.nodes.keys()) + 1
        elif node_id in self.nodes:
            raise ValueError(f"Node ID {node_id} already exists")
        
        # Create and store the node
        node = Node(node_id, np.array([x, y, z]))
        self.nodes[node_id] = node
        
        return node
    
    def add_nodes(self, coords: list[list[float]]) -> None:
        """Add nodes to the frame."""
        for coord in coords: 
            self.add_node(coord[0], coord[1], coord[2])
    
    def add_material(self, name: str, material: Material) -> None:
        """
        Add a material to the frame. 
        """
        if name in self.materials:
            raise ValueError(f"Material '{name}' already exists")
        
        self.materials[name] = material
        
    def add_section(self, name: str, section: Section) -> None:
        """
        Add a cross-section to the frame.
        """
        if name in self.sections:
            raise ValueError(f"Section '{name}' already exists")
        self.sections[name] = section
            
    def add_element(self, node_ids: List[int], material_name: str, section_name: str, element_class: Element = Element, element_id: int=None) -> Element:
        """
        Add an element to the frame, connecting specified nodes with given properties.
        """
        # Find the nodes by ID
        element_nodes = []
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in the frame")
            element_nodes.append(self.nodes[node_id])
        
        # Get the material and section
        if material_name not in self.materials:
            raise KeyError(f"Material '{material_name}' not found")
        material = self.materials[material_name]
        
        if section_name not in self.sections:
            raise KeyError(f"Section '{section_name}' not found")
        section = self.sections[section_name]
        
        # Assign an ID to the element
        if element_id is None:
            # Find the next available ID
            element_id = 0 if not self.elements else max(self.elements.keys()) + 1
        elif element_id in self.elements:
            raise ValueError(f"Element ID {element_id} already exists")
        
        element = element_class(element_id, element_nodes, material, section)
        
        # Add the element to the dictionary
        self.elements[element_id] = element
        
        return element
    
    def add_elements(self, node_ids_list: List[list[int]], material_name: str, section_name: str, element_class=Element, element_id=None) -> None:
        """Add elements to the frame."""
        for node_ids in node_ids_list: 
            self.add_element(node_ids, material_name, section_name, element_class, element_id)
        
    def add_boundary_condition(self, node_id: int, constraints: List[bool], boundary_class) -> None:
        """Add a boundary condition to a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in the frame")
        
        self.boundary_conditions[node_id] = boundary_class(node_id, constraints)
        
    def add_nodal_load(self, node_id: int, forces: List[float], load_class) -> None:
        """Add a load to a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in the frame")
        
        self.loads[node_id] = load_class(node_id, forces)

    def solve(self, solver_type: str='direct') -> None:
        """
        Solve the frame model and return results.
    
        """
        assembler = Assembler(self)
        
        # Create and run solver
        solver = LinearSolver(solver_type=solver_type)
        nodal_displacements, nodal_forces = solver.solve(assembler)
        results = Results(assembler, nodal_displacements, nodal_forces)
        
        # Store results in the frame 
        self.results = results
        return results 
    
    def show(self, scale: float=1.0, show_undeformed: bool=True) -> None:
        """
        Plot the deformed shape of the frame.
        
        """
        if self.results is None:
            print("Model has not been solved yet. Solving with default settings...")
            self.solve()
        
        return self.results.plot_deformed_shape(
            scale=scale, 
            show_undeformed=show_undeformed
        )